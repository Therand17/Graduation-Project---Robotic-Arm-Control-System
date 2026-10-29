import os
import sys
import warnings
import numpy as np
import joblib
import mne
import pygame
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from robot_simulator import SimpleHand
from real_improved_csp import RobustCSP
import params
from edf import getAllEpochFormatedData,getAllEpochFormatedData_advanced
from sklearn.svm import SVC 
FS = params.BASE_SAMPLES_PER_SECONDE
MOTOR_CHANNELS = list(range(12)) 

import serial
import time

def init_environment():
    os.makedirs(params.MODEL_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")

def convert_session_to_runs(sessions):
    session_to_run_map = {
        1: 1, 
        2: 2, 
        3: 3,  
        4: 4,   
        5: 5,   
        6:6,
        7: 7,   
        8: 8,    
        11: 11,
        12: 12,
    }
    
    runs = []
    for session in sessions:
        if session in session_to_run_map:
            runs.append(session_to_run_map[session])
        else:
            print(f"Warning: Session {session} not in map, using as is")
            runs.append(session)
    
    return runs


class ArduinoController:
    def __init__(self, port="COM3", baud=9600):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            print("[ARDUINO] Connected successfully.")
        except:
            print("[ARDUINO] Connection failed.")
            self.ser = None

    def open_hand(self):
        if self.ser:
            self.ser.write(b"ANGLE:80\n")

    def close_hand(self):
        if self.ser:
            self.ser.write(b"ANGLE:0\n")

    def close(self):
        if self.ser:
            self.ser.close()

def scale_and_filter_data(X, fs, scale_factor=1e6, low_freq=8, high_freq=30):
    """Scaling and filtering"""
    X_scaled = X * scale_factor
    
    nyq = fs / 2.0
    low_norm = max(low_freq / nyq, 1e-6)
    high_norm = min(high_freq / nyq, 0.999999)
    b, a = butter(4, [low_norm, high_norm], btype='band')
    X_filtered = filtfilt(b, a, X_scaled, axis=-1)
    
    X_car = X_filtered - np.mean(X_filtered, axis=1, keepdims=True)
    
    return X_car

class RobotArmControl:
    """
    Robot arm control logic.
    Uses Buffer to reduce noise and Toggle to switch between states.
    """
    def __init__(self, buffer_size=5, threshold=0.5, refractory_period_sec=2.0):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.refractory_period = refractory_period_sec
        
        # State
        self.prediction_buffer = []
        self.last_toggle_time = time.time()  # Real time at start
        self.is_arm_open = False
        
    def update(self, prediction_label):
        """
        Update state based on prediction (0=Rest, 1=Left MI).
        Uses system real time
        """
        # 1. Add to Buffer
        self.prediction_buffer.append(prediction_label)
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
        
        action_taken = "HOLD"
        
        # 2. Make decision if Buffer is full
        if len(self.prediction_buffer) == self.buffer_size:
            current_time = time.time()  # Real time
            time_since_last_toggle = current_time - self.last_toggle_time
            
            # Check Refractory Period
            if time_since_last_toggle > self.refractory_period:
                
                # Calculate ratio of motion signals
                activity_ratio = sum(self.prediction_buffer) / self.buffer_size
                
                if activity_ratio >= self.threshold:
                    action_taken = self.toggle_arm()
                    self.last_toggle_time = current_time  # Update with real time
                    self.prediction_buffer = []
        
        return action_taken

    def toggle_arm(self):
        """Execute Toggle command"""
        self.is_arm_open = not self.is_arm_open
        status = "OPEN" if self.is_arm_open else "CLOSE"
        return f"TOGGLE_{status}"

def train_cross_session_robust(subject, train_sessions, test_sessions):
    
    print(f"\n{'='*80}")
    print(f"CROSS-SESSION ROBUST MODEL (Rest vs Left) | Subject {subject:03d}")
    
    # Convert sessions to Runs
    train_runs = convert_session_to_runs(train_sessions)
    test_runs = convert_session_to_runs(test_sessions)
    
    print(f"Train sessions: {train_sessions} -> Runs: {train_runs}")
    print(f"Test sessions: {test_sessions} -> Runs: {test_runs}")
    print('='*80)
  
    # 1. Collect data using advanced function
    print("[1/6] Collecting training and test data using Advanced Function...")
    
    if not test_runs:
        print("ERROR: No test sessions provided!")
        return None
    
    target_test_run = test_runs[0]
    
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []
    
    try:
        # Call advanced function
        X_train_data, y_train_data, X_test_data, y_test_data = getAllEpochFormatedData_advanced(subject, test_run=target_test_run)
        
        X_train_all.append(X_train_data)
        y_train_all.append(y_train_data)
        
        X_test_all.append(X_test_data)
        y_test_all.append(y_test_data)
        
        print(f"   Loaded successfully.")
        print(f"   Training Data: {len(y_train_data)} trials (Rest={sum(y_train_data==0)}, Left={sum(y_train_data==1)})")
        print(f"   Test Data: {len(y_test_data)} trials (Rest={sum(y_test_data==0)}, Left={sum(y_test_data==1)})")
        
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    if not X_train_all or not X_test_all:
        print("ERROR: No data found!")
        return None
    
    # Merge data
    X_train = np.vstack(X_train_all)
    y_train = np.concatenate(y_train_all)
    X_test = np.vstack(X_test_all)
    y_test = np.concatenate(y_test_all)
    
    print(f"   Total training data: {X_train.shape[0]} trials")
    print(f"   Total test data: {X_test.shape[0]} trials")
    
    # 2. Select channels
    print("\n[2/6] Selecting motor channels...")
    X_train = X_train[:, MOTOR_CHANNELS, :]
    X_test = X_test[:, MOTOR_CHANNELS, :]
    print(f"   Using {len(MOTOR_CHANNELS)} motor channels")
    
    # 3. Scaling and filtering
    print("\n[3/6] Scaling and filtering...")
    X_train_processed = scale_and_filter_data(X_train, FS)
    X_test_processed = scale_and_filter_data(X_test, FS)
    
    # 4. Select best CSP with Cross-Validation
    print("\n[4/6] Finding best CSP configuration with Cross-Validation...")
    best_score = 0
    best_csp = None
    best_n_components = 4
    best_reg = 0.1
    
    cv_scores_all = []  # To store all CV results
    
    for n_components in [2, 4, 8]:
        for reg in [0.005, 0.1, 0.2]:
            csp = RobustCSP(n_components=n_components, reg=reg, log=True)
            
            try:
                temp_scores = []
                cv_temp = StratifiedKFold(n_splits=min(5, len(y_train)//5), shuffle=True, random_state=42)
                
                for tr_idx, val_idx in cv_temp.split(X_train_processed, y_train):
                    X_tr, X_val = X_train_processed[tr_idx], X_train_processed[val_idx]
                    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                    
                    if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                        continue
                    
                    X_tr_csp = csp.fit_transform(X_tr, y_tr)
                    X_val_csp = csp.transform(X_val)
                    
                    scaler = StandardScaler()
                    X_tr_scaled = scaler.fit_transform(X_tr_csp)
                    X_val_scaled = scaler.transform(X_val_csp)
                    
                    # === Modification: Using SVM with class_weight='balanced' ===
                    clf = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
                    clf.fit(X_tr_scaled, y_tr)
                    temp_scores.append(clf.score(X_val_scaled, y_val))
   
                
                if temp_scores:
                    mean_score = np.mean(temp_scores)
                    std_score = np.std(temp_scores)
                    cv_scores_all.append((n_components, reg, mean_score, std_score))
                    
                    print(f"   CSP (n={n_components}, reg={reg}): {mean_score:.4f} ± {std_score:.4f}")
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_csp = csp
                        best_n_components = n_components
                        best_reg = reg
                        
            except Exception as e:
                print(f"   CSP (n={n_components}, reg={reg}) failed: {e}")
    
    if best_csp is None:
        print("   Warning: Using default CSP configuration")
        best_csp = RobustCSP(n_components=4, reg=0.1, log=True)
        best_n_components = 4
        best_reg = 0.1
    
    print(f"   Selected: CSP with {best_n_components} components, reg={best_reg}")
    
    # 5. Final Training
    print("\n[5/6] Training final model...")
    
    # Apply CSP to training data
    X_train_csp = best_csp.fit_transform(X_train_processed, y_train)
    
    # Scaling
    scaler_final = StandardScaler().fit(X_train_csp)
    X_train_scaled = scaler_final.transform(X_train_csp)
    
    # === Modification: Final training using SVM ===
    # No need for manual balancing, SVM handles it via class_weight
    clf_final = SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced', random_state=42)
    clf_final.fit(X_train_scaled, y_train)

    
    # Evaluate on training data
    train_acc = clf_final.score(X_train_scaled, y_train)
    
    # 6. Evaluate on test data
    print("\n[6/6] Evaluating on test sessions...")
    
    # Apply CSP to test data
    X_test_csp = best_csp.transform(X_test_processed)
    X_test_scaled = scaler_final.transform(X_test_csp)
    
    # Prediction
    y_pred = clf_final.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    
    # 7. Display Results
    print(f"\n{'='*80}")
    print("CROSS-SESSION RESULTS:")
    print('='*80)
    print(f"Training sessions: {train_sessions} -> Runs: {train_runs}")
    print(f"Test sessions: {test_sessions} -> Runs: {test_runs}")
    print(f"\nTraining Data: {X_train.shape[0]} trials")
    print(f"Test Data: {X_test.shape[0]} trials")
    
    # Print best CV result
    if cv_scores_all:
        best_cv_score = max(cv_scores_all, key=lambda x: x[2])
        print(f"\nBest Cross-Validation Score: {best_cv_score[2]:.4f} ± {best_cv_score[3]:.4f}")
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Detailed report
    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, y_pred, target_names=['Rest', 'Left Hand']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix (Test):")
    print(f"               Predicted")
    print(f"               Rest   Left")
    print(f"Actual Rest    {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"       Left    {cm[1,0]:3d}     {cm[1,1]:3d}")
    
    # Quality Assessment
    if test_acc >= 0.75:
        status = "EXCELLENT"
    elif test_acc >= 0.65:
        status = "GOOD"
    elif test_acc >= 0.55:
        status = "FAIR"
    else:
        status = "POOR"
    
    print(f"\nStatus: {status} (Chance level: 0.50)")
    print('='*80)
    
    # 8. Save Model
    model_path = os.path.join(params.MODEL_DIR, f"subj{subject:03d}_robot_arm.joblib")
    joblib.dump({
        'csp': best_csp,
        'scaler': scaler_final,
        'clf': clf_final,
        'channels': MOTOR_CHANNELS,
        'train_sessions': train_sessions,
        'train_runs': train_runs,
        'test_sessions': test_sessions,
        'test_runs': test_runs,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_scores': cv_scores_all,
        'csp_config': {
            'n_components': best_n_components,
            'reg': best_reg
        },
        'data_info': {
            'n_train_trials': len(y_train),
            'n_test_sessions': len(y_test),
            'subject': subject
        }
    }, model_path)
    
    print(f"\n[SAVED] Robot Arm model saved to: {model_path}")
    print('='*80)
    
    return test_acc


# Prediction and Control (Control Loop) 
def predict_and_control(subject, session, use_simulator=False, use_arduino=False):
    """
    Load data, execute predictions with real time,
    then run simulation.
    """
    
    # Load model
    model_path = os.path.join(params.MODEL_DIR, f"subj{subject:03d}_robot_arm.joblib")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found. Train first!")
        return
    
    model = joblib.load(model_path)
    csp = model['csp']
    scaler = model['scaler']
    clf = model['clf']
    train_accuracy = model['train_accuracy']
    train_sessions = model['train_sessions']
    train_runs = model['train_runs']
    
    # Load data
    runs = convert_session_to_runs([session])
    run = runs[0]
    
    print(f"\n{'='*80}")
    print(f"PROCESSING & CONTROL | Subject {subject:03d} | Session {session} (Run {run})")
    print('='*80)
    
    # 1. Load and process data
    print("[1/4] Loading EEG Data...")
    try:
        X_raw, y_true = getAllEpochFormatedData(subject, run)
        X_raw = np.asarray(X_raw)[:, MOTOR_CHANNELS, :]
        n_trials = X_raw.shape[0]
        print(f"   Loaded {n_trials} trials.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("[2/4] Preprocessing & Feature Extraction...")
    X_processed = scale_and_filter_data(X_raw, FS)
    X_csp = csp.transform(X_processed)
    X_scaled = scaler.transform(X_csp)
    
    # 3. Calculate Cross-Session Accuracy
    print("\n[3/4] Computing Cross-Session Accuracy...")
    y_pred_all = clf.predict(X_scaled)
    test_accuracy = accuracy_score(y_true, y_pred_all)
    
    # Print reports...
    print(f"\n{'='*80}")
    print("CROSS-SESSION ACCURACY COMPARISON")
    print('='*80)
    print(f"Training Sessions: {train_sessions} (Runs: {train_runs})")
    print(f"Test Session: {session} (Run: {run})")
    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

    accuracy_diff = test_accuracy - train_accuracy
    if accuracy_diff > 0:
        diff_str = f"+{accuracy_diff:.4f}"
        diff_status = "BETTER than training"
    elif accuracy_diff < 0:
        diff_str = f"{accuracy_diff:.4f}"
        diff_status = "WORSE than training"
    else:
        diff_str = "0.0000"
        diff_status = "SAME as training"
    

    
    # Print classification report
    print(f"\nClassification Report (Cross-Session Test):")
    print(classification_report(y_true, y_pred_all, target_names=['Rest', 'Left Hand']))
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred_all)
    print(f"Confusion Matrix (Test):")
    print(f"               Predicted")
    print(f"               Rest   Left")
    print(f"Actual Rest    {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"       Left    {cm[1,0]:3d}     {cm[1,1]:3d}")
    
    # Calculate accuracy per class
    if cm[0,0] + cm[0,1] > 0:
        rest_acc = cm[0,0] / (cm[0,0] + cm[0,1])
        print(f"\nRest Accuracy:     {rest_acc:.3f}")
    if cm[1,0] + cm[1,1] > 0:
        left_acc = cm[1,1] / (cm[1,0] + cm[1,1])
        print(f"Left Hand Accuracy: {left_acc:.3f}")
    
    # Cross-Session quality assessment
    if test_accuracy >= 0.75:
        status = "EXCELLENT Cross-Session Transfer"
    elif test_accuracy >= 0.65:
        status = "GOOD Cross-Session Transfer"
    elif test_accuracy >= 0.55:
        status = "FAIR Cross-Session Transfer"
    else:
        status = "POOR Cross-Session Transfer"
    
    print(f"\nCross-Session Status: {status}")
    print('='*80)
    

    print("\n[4/4] Running Real-time Predictions & Simulation...")

    # Initialize Arduino if requested

    arduino = None
    if use_arduino:
        print("[ARDUINO] Initializing connection...")
        try:
            import serial
            arduino = serial.Serial("COM3", 9600, timeout=1)  # <-- غير COM3 حسب جهازك
            time.sleep(2)
            print("[ARDUINO] Connected successfully.")
        except Exception as e:
            print(f"[ARDUINO] Connection failed: {e}")
            use_arduino = False
    
    controller = RobotArmControl(buffer_size=5, threshold=0.5, refractory_period_sec=2.0)
    
    # Initialize simulator directly if requested
    sim = None
    if use_simulator:
        print("[SIM] Initializing Real-time Hand Simulator...")
        try:
            pygame.init()
            WIDTH, HEIGHT = 1000, 700
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            pygame.display.set_caption(f"EEG BCI REAL-TIME - Subj {subject} | Session {session}")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont("Arial", 24)
            sim = SimpleHand()
            sim_running = True
        except Exception as e:
            print(f"Could not initialize simulator: {e}")
            use_simulator = False
    
    # List to store events
    timeline_events = []
    
    # Start time
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"{'Trial':<8} | {'Raw':<5} | {'Buffer':<10} | {'Action':<12} | {'Arm State':<10} | {'Time':<10}")
    print("-" * 90)
    
    for i in range(n_trials):
        # Update simulator in each cycle
        if use_simulator and sim_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sim_running = False
                elif event.type == pygame.VIDEORESIZE:
                    WIDTH, HEIGHT = event.size
                    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sim_running = False
        
        # Raw prediction
        raw_pred = int(clf.predict([X_scaled[i]])[0])
        
        # Update control logic
        action = controller.update(raw_pred)
        
        # Update simulator directly when any action is executed
        if use_simulator and sim:
            if action == "TOGGLE_OPEN":
                sim.set_state("open")
                print(f"   REAL-TIME: Hand OPEN at trial {i+1}")
            elif action == "TOGGLE_CLOSE":
                sim.set_state("closed")
                print(f"   REAL-TIME: Hand CLOSED at trial {i+1}")

        
        # Send command to Arduino
        
        if use_arduino and arduino:
            try:
                # عكس الاتجاه
                if action == "TOGGLE_OPEN":
                    arduino.write(b"ANGLE:0\n")     # فتح فعلي
                elif action == "TOGGLE_CLOSE":
                    arduino.write(b"ANGLE:80\n")    # غلق فعلي
            except Exception as e:
                print(f"[ARDUINO] Write error: {e}")        
        
        # Log state
        arm_state = "OPEN" if controller.is_arm_open else "CLOSED"
        buffer_str = "".join(map(str, controller.prediction_buffer)).ljust(5)
        elapsed_time = time.time() - start_time
        
        # Print line
        print(f"{i+1:<8} | {raw_pred:<5} | {buffer_str:<10} | {action:<12} | {arm_state:<10} | {elapsed_time:.2f}s")
        
        # Store event
        timeline_events.append({
            'time': elapsed_time,
            'state': arm_state,
            'action': action,
            'trial_idx': i
        })

        time.sleep(0.1)
        # Draw simulator in each cycle
        if use_simulator and sim and sim_running:
            screen.fill((245, 247, 250))
            
            # Draw hand
            sim.draw(screen, WIDTH, HEIGHT)
            
            # Draw info
            info_rect = pygame.Rect(20, 20, 350, 200)
            pygame.draw.rect(screen, (255, 255, 255), info_rect)
            pygame.draw.rect(screen, (0, 0, 0), info_rect, 2)
            
            title_font = pygame.font.SysFont("Arial", 28, bold=True)
            title_surf = title_font.render(f"Subject {subject} - Session {session}", True, (50, 50, 50))
            screen.blit(title_surf, (30, 30))
            
            time_text = font.render(f"Time: {elapsed_time:.2f}s", True, (0, 0, 150))
            screen.blit(time_text, (30, 70))
            
            trial_text = font.render(f"Trial: {i+1}/{n_trials}", True, (0, 0, 0))
            screen.blit(trial_text, (30, 100))
            
            state_color = (0, 150, 0) if arm_state == "OPEN" else (200, 0, 0)
            state_text = font.render(f"Arm State: {arm_state}", True, state_color)
            screen.blit(state_text, (30, 130))
            
            action_text = font.render(f"Last Action: {action}", True, (100, 100, 100))
            screen.blit(action_text, (30, 160))
            
            pred_text = font.render(f"Prediction: {'Left' if raw_pred else 'Rest'}", True, (0, 0, 0))
            screen.blit(pred_text, (30, 190))
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS
        
        # Short wait to simulate real time
        time.sleep(0.1) # يوقف البرنامج 0.1 ثانية عشان يحافظ على الإحساس بالوقت الحقيقي (Real-time)
    
    print("="*80)
    print("Real-time Prediction & Simulation Finished.") # طبع خط فاصل + رسالة انتهاءويعني خلصت كل التوقعات
    toggle_count = sum(1 for e in timeline_events if "TOGGLE" in e['action']) # يمر على كل الاحداث و يحسب كم مره قتح و قفل 
    print(f"   Total Actions (Toggles): {toggle_count}")
    print(f"   Final Arm State: {timeline_events[-1]['state']}") # عدد حركات اليد و اخر حالة لليد
    print("="*80)
    
    # Keep simulation open after finishing
    if use_simulator and sim and sim_running:  # اذا المحاكاة شغاله ماتقفل مباشره 
        print("\n[SYSTEM] Simulation completed. Press ESC or close window to exit.")  # ESC or close window to exit
        while sim_running: # lOOP Watting شغال لين المستخدم يقفله
            for event in pygame.event.get(): # قراءة أحداث المستخدم
                if event.type == pygame.QUIT: #حالات الاغلاق يا اغلاق نافذه او 
                    sim_running = False
                elif event.type == pygame.KEYDOWN: # ضغط زر 
                    if event.key == pygame.K_ESCAPE: # زر ESC
                        sim_running = False
            
            # Continue displaying last state
            screen.fill((245, 247, 250))
            sim.draw(screen, WIDTH, HEIGHT) # يرسم اليد بنفس حالتها الأخيرة
            
            # Draw completion message
            font_large = pygame.font.SysFont("Arial", 32, bold=True)
            complete_text = font_large.render("SIMULATION COMPLETED", True, (0, 150, 0)) # يكتب الرساله الاخيره SIMULATION COMPLETED
            text_rect = complete_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(complete_text, text_rect)
            
            exit_text = font.render("Press ESC to exit", True, (100, 100, 100)) #رسالة الخروج 
            exit_rect = exit_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 50))
            screen.blit(exit_text, exit_rect)
            
            pygame.display.flip() # تحديث الشاشة كل 30 فريم
            clock.tick(30)
        
        pygame.quit()
        print("[SYSTEM] Simulator closed.") # اغلاق المحاكاة 
        
    # Close Arduino connection safely
    if use_arduino and arduino:
        arduino.close() #Close Arduino
        print("[ARDUINO] Connection closed.")    
    
    return { # ارجاع النتايج 
        'train_accuracy': train_accuracy, # دقة التدريب 
        'test_accuracy': test_accuracy, # دقة الاختبار 
        'accuracy_diff': test_accuracy - train_accuracy,# الفرق بينهم
        'n_toggles': toggle_count, # عدد الحركات 
        'final_state': timeline_events[-1]['state'] if timeline_events else None # الحالة النهاىيه 
    }



if __name__ == "__main__": # الجزء الاخير تشغيل البرنامج
    init_environment()
    
    if len(sys.argv) < 3:
        print("Usage: python main.py <subject_id> <mode> [args...]")
        sys.exit(1)
    # قراءة input
    subject = int(sys.argv[1]) # رقم الشخص 
    remaining_args = sys.argv[2:] # باقي الاوامر
    
    # Initialize variables
    mode = None  # وضعين للمود'train' or 'control' 
    train_sessions = [] # تدريب المودل 
    test_sessions = [] # تشغيل النظام
    control_session = None 
    
    # NEW FLAGS
    use_simulator = False
    use_arduino = False
    
    # Search for mode and options 
    for arg in remaining_args:
        arg_lower = arg.lower()
        
        if arg_lower in ['train', 'control']:
            mode = arg_lower
        
        # Simulator + Arduino تشغيل الاثنين 
        elif arg_lower == '-s':
            use_simulator = True
            use_arduino = True
        
        # Simulator only محاكاة
        elif arg_lower == '--sim':
            use_simulator = True
            use_arduino = False
        
        # Arduino only
        elif arg_lower == '--simulator':
            use_simulator = False
            use_arduino = True


    
    if mode is None:
        print("Error: Mode not specified. Use 'train' or 'control'")
        sys.exit(1)
    
    filtered_args = [] #فلترة اللاوامر حذف الكلمات غير المهمة
    for arg in remaining_args:
        arg_lower = arg.lower()
        if arg_lower not in [
            'train', 'control',
            '--sim', '--simulator', '-s',
            '--loop', '-l', 'loop'
        ]:
            filtered_args.append(arg)
    

    if mode == 'train': #وضع التدريب 
        
        if len(filtered_args) == 1 and filtered_args[0].isdigit():# إذا المستخدم أدخل قيمة واحدة فقط (جلسة تدريب واحدة)
            train_sessions = [int(filtered_args[0])]   # تحويلها إلى قائمة
            test_sessions = [] # ما فيه اختبار حالياً
        
        elif len(filtered_args) >= 2: # إذا المستخدم أدخل قيمتين أو أكثر (تدريب + اختبار)
            
            if ',' in filtered_args[0]: # إذا جلسات التدريب فيها فاصلة (مثلاً "3,4,5")
                train_sessions = list(map(int, filtered_args[0].split(',')))  # نحول النص إلى قائمة أرقام
            else:
                train_sessions = [int(filtered_args[0])]    # إذا رقم واحد فقط
            
            if ',' in filtered_args[1]:  # نفس الشيء لجلسات الاختبار
                test_sessions = list(map(int, filtered_args[1].split(',')))
            else:
                test_sessions = [int(filtered_args[1])]
        
        else:
            print("Error: Invalid training arguments format")     # إذا الإدخال غير صحيح
            sys.exit(1)

        if not train_sessions: # التأكد من وجود جلسات تدريب
            print("No training sessions")
            sys.exit(1)
        
        if not test_sessions:   # إذا ما تم تحديد جلسات اختبار
            if len(train_sessions) > 1: # إذا عندنا أكثر من جلسة تدريب
                test_sessions = [train_sessions[-1]]  # نأخذ آخر جلسة كاختبار
                train_sessions = train_sessions[:-1]  # والباقي للتدريب
            else:
                test_sessions = train_sessions.copy() # إذا جلسة واحدةالتدريب = الاختبار
        
        train_cross_session_robust(subject, train_sessions, test_sessions)  # يدرب المودل 
    

    elif mode == 'control': # تشغيل النظام
        
        if len(filtered_args) >= 1 and filtered_args[0].isdigit():  # إذا أول قيمة رقم هي الجلسة
            control_session = int(filtered_args[0])
        else:
            for arg in filtered_args:   # إذا لا، نبحث عن أول رقم داخل القائمة
                if arg.isdigit():
                    control_session = int(arg)
                    break
        
        if control_session is None: # إذا ما وجدنا جلسة
            print("Error: Control mode requires a session number")
            sys.exit(1)
        # عرض معلومات التشغيل
        print(f"Control mode: Subject {subject}")
        print(f"   Session: {control_session}")
        print(f"   Use Simulator: {'Yes' if use_simulator else 'No'}")
        print(f"   Use Arduino  : {'Yes' if use_arduino else 'No'}")
        
        # Execute control with both flags تشغيل التنبؤ + التحكم + المحاكاة
        predict_and_control(
            subject,
            control_session,
            use_simulator=use_simulator,
            use_arduino=use_arduino
        )
    # حالة خطأ (mode غير صحيح
    else:
        print("Error: Invalid mode")
        sys.exit(1)