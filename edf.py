import numpy as np
import mne
from mne import Epochs, events_from_annotations
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import params
import utils

TMIN = params.TMIN
TMAX = params.TMAX
L_FREQ = params.L_FREQ
H_FREQ = params.H_FREQ
BASE_SAMPLES_PER_SECONDE = params.BASE_SAMPLES_PER_SECONDE
# PICK_CHANNELS = params.PICK_CHANNELS
RUNS_MI_HAND = params.RUNS_MI_HAND

getRawEDF = utils.getRawEDF


def preprocess_raw(raw: mne.io.BaseRaw):
  
    print("[PREPROCESS] Applying Band-pass Filter (0.5 - 40 Hz)...")
    raw.filter(L_FREQ, H_FREQ, fir_design='firwin', skip_by_annotation='edge')

    print("[PREPROCESS] Applying Arm Control Logic (T1 Only)...")
    raw.notch_filter(freqs=[50, 70])

    print("[PREPROCESS] Applying Common Average Reference (CAR)...")
    raw.set_eeg_reference('average', projection=True)

    # print("[PREPROCESS] Applying ICA for Artifact Removal...")
    # ica = mne.preprocessing.ICA(n_components=15, random_state=97)
    # ica.fit(raw)
    # raw = ica.apply(raw)
    
    return raw


# Epoch extraction
def getEpoch(raw: mne.io.BaseRaw, tmin, tmax, label):
    events, mapping = events_from_annotations(raw, verbose=False)
    
    if label not in mapping:
        # Flexible search for the label
        possible_keys = [k for k in mapping.keys() if label.strip('.') in k]
        if possible_keys:
            label = possible_keys[0]
        else:
            raise ValueError(f"No {label} annotations found. Available: {mapping}")

    epochs = Epochs(
        raw,
        events,
        event_id={label: mapping[label]},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    )
    
    if len(epochs) == 0:
        raise ValueError(f"No usable {label} epochs.")
    return epochs


def extractEdfEpochs(filepath):
    """
    Modified for Rest vs Left Hand (T0 vs T1) with Dynamic Channel Selection.
    """
    raw = getRawEDF(filepath)
    raw_clean = preprocess_raw(raw)
    
    try:
        raw_clean.pick_channels(params.MOTOR_CHANNEL_NAMES)
        
        current_names = raw_clean.ch_names
        desired_names = params.MOTOR_CHANNEL_NAMES
        
        if current_names != desired_names:
            raw_clean.reorder_channels(desired_names)
            
        print(f"[INFO] Selected {len(current_names)} motor channels in correct order.")
        
    except ValueError as e:
        print(f"[ERROR] Could not pick desired channels: {e}")
        print(f"        Available: {raw_clean.ch_names}")
        print(f"        Requested: {params.MOTOR_CHANNEL_NAMES}")

    print(f"[DEBUG] Channels after picking: {raw_clean.ch_names}")

    try:
        # Extract T0 (Rest)
        ep_t0 = getEpoch(raw_clean, TMIN, TMAX, "T0")
        # Extract T1 (Left Hand)
        ep_t1 = getEpoch(raw_clean, TMIN, TMAX, "T1")
        
        # print statements...
        
    except ValueError as e:
        print(f"[ERROR] {e}")
        return {0: [], 1: []}

    X_rest = list(ep_t0.get_data())
    X_left = list(ep_t1.get_data())

    return {0: X_rest, 1: X_left}

def getAllEpochFormatedData(subj, test_run=None):

    X, y = [], []
    
    runs_to_use = RUNS_MI_HAND
    if test_run is not None and test_run in runs_to_use:
        runs_to_use = [r for r in runs_to_use if r != test_run]
        print(f"[INFO] Training on Runs: {runs_to_use} (Excluding {test_run})")
    else:
        print(f"[ MIS-TYPED VARIABLE] {RUNS_MI_HAND} vs {runs_to_use}")
        print(f"[INFO] Training on ALL Runs: {runs_to_use}")

    for run in runs_to_use:
        try:
            epochs = extractEdfEpochs(utils.getEdfFilePath(subj, run))
            
            X.extend(epochs[0])
            y.extend([0] * len(epochs[0]))

            X.extend(epochs[1])
            y.extend([1] * len(epochs[1]))
        except Exception as e:
            print(f"[WARNING] Skipping Run {run}: {e}")
            continue

    if not X:
        raise RuntimeError("No epochs extracted from any run")

    print(f"[INFO] Total Data Collected: {len(X)} trials")
    return X, y

def getAllEpochFormatedData_advanced(subj, test_run=None, augment_data=False):

    print(f"[ADVANCED DATA COLLECTION] Subject {subj:03d}")
    print("-" * 60)
    
    # Define Motor Imagery Runs
    ALL_MI_RUNS = [3,4,7,8, 11, 12]
    
    available_runs = []
    for run in ALL_MI_RUNS:
        if os.path.exists(utils.getEdfFilePath(subj, run)):
            available_runs.append(run)
    
    if not available_runs:
        raise FileNotFoundError(f"No data for subject {subj}")
    
    if test_run is not None and int(test_run) in available_runs:
        train_runs = [r for r in available_runs if r != int(test_run)]
        test_runs = [int(test_run)]
    else:
        train_runs = available_runs
        test_runs = []
    
    X_train, y_train = [], []
    for run in train_runs:
        try:
            epochs_dict = extractEdfEpochs(utils.getEdfFilePath(subj, run))
            
            for class_label in [0, 1]:
                trials = epochs_dict.get(class_label, [])
                X_train.extend(trials)
                y_train.extend([class_label] * len(trials))
                
                if augment_data and len(trials) < 30:
                    augmented = augment_eeg_trials(trials)
                    X_train.extend(augmented)
                    y_train.extend([class_label] * len(augmented))
                    
        except Exception as e:
            print(f"Warning: Could not process Run {run}: {e}")
            continue
    
    X_test, y_test = [], []
    if test_runs:
        for run in test_runs:
            try:
                epochs_dict = extractEdfEpochs(utils.getEdfFilePath(subj, run))
                
                for  label in [0, 1]:
                    trials = epochs_dict.get(label, [])
                    X_test.extend(trials)
                    y_test.extend([label] * len(trials))
                    
            except Exception as e:
                print(f"Warning: T0 vs T1 Error in Run {run}: {e}")
                continue
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if len(X_test) > 0:
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train

def augment_eeg_trials(trials, num_augmented=5):
    if not trials:
        return []
    
    augmented = []
    for _ in range(num_augmented):
        for trial in trials:
            # 1. Add Gaussian noise
            noise = np.random.normal(0, 0.01, trial.shape)
            trial_noisy = trial + noise
            
            # 2. Small scaling
            scale = np.random.uniform(0.95, 1.05)
            trial_scaled = trial_noisy * scale
            
            # 3. Small time shift
            shift = np.random.randint(-2, 3) 
            if shift > 0:
                trial_shifted = np.pad(trial_scaled[:, shift:], ((0, 0), (0, shift)), mode='edge')
            elif shift < 0:
                trial_shifted = np.pad(trial_scaled[:, :shift], ((0, 0), (-shift, 0)), mode='edge')
            else:
                trial_shifted = trial_scaled
            
            augmented.append(trial_shifted)
    
    return augmented