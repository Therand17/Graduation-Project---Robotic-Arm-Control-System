import os  #للتعامل مع الملفات والمسارات
import sys  #للتعامل مع الملفات والمسارات
import joblib   #لتحميل المودل
import mne    
from mne.io.base import BaseRaw
import numpy as np                              #import Libraries

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import params

MODEL_DIR = params.MODEL_DIR


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")     #تحديد المشروع + Data set

def getEdfFilePath(subj, run):  # مسار ملف EEG , subj = رقم الشخص , run = رقم الجلسة
    filename = f"S{subj:03d}R{run:02d}.edf"
    path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"EDF file not found: {path}")  # اذا الملف ماهو موجود يعطي خطا 

    return path


def getRawEDF(filepath) -> BaseRaw:  #قرات ملف EEG 
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False) # MNE تحويل الملف الى بيانات جاهزه لتحليل 
    return raw



def loadModel(subj, run):  # تحميل الملف المدرب 
    model_file = os.path.join(
        MODEL_DIR,
        f"subj{subj:03d}_run{run:02d}.joblib" #يبحث عن المودل 
    )

    if not os.path.exists(model_file):
        model_file_new = os.path.join(
            MODEL_DIR,
            f"subj{subj:03d}_run{run:02d}_csp_svm.joblib" # اذا مالقى يجرب 
        )
        if os.path.exists(model_file_new):
            return joblib.load(model_file_new)
            
        raise FileNotFoundError(f"Model not found: {model_file}") # اذا مالقى الاثنين 

    return joblib.load(model_file) # اذا لقى يرجع المودل جاهز للاستخدام 

