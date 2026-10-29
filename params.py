

TMIN = 0.0          # Epoch start (seconds)
TMAX = 2.0          # Epoch end (seconds), نحلل الإشارة من 0 إلى 2 ثانية
L_FREQ = 0.5        # Band-pass low cutoff (Hz)
H_FREQ = 40       # Band-pass high cutoff (Hz) , حتفظ فقط بالترددات بين 0.5 -40  
BASE_SAMPLES_PER_SECONDE = 160 #

# Number of CSP components
# N_COMPONENTS = 6
N_COMPONENTS = 6 #عدد الخصائص (features) المستخرجة من الإشارة

# Directory to save trained models
MODEL_DIR = 'models' #كل المودلات محفوظة هنا
# 
# RUNS_MI_HAND = [4, 8, 12]
RUNS_MI_HAND = [3,4,5,7, 8, 11, 12] # التجارب الخاصة بحركة اليد

# PICK_CHANNELS = [
#     'Fc3.', 'Fc4.', 'Fcz.', 
#     'C3..', 'C4..', 'Cz..', 
#     'Cp3.', 'Cp4.', 'Cpz.'  
# ]

MOTOR_CHANNEL_NAMES = [
    'Fc3.', 'Fc4.', 'Fcz.', 
    'C3..', 'C4..', 'Cz..', #أسماء قنوات الدماغ المرتبطة بالحركة
    'Cp3.', 'Cp4.', 'Cpz.',
    'P3..', 'P4..', 'Pz..'  
]

