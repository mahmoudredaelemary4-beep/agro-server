import os
from tensorflow.keras.models import load_model

# تأكد إن ملف الموديل جنبه
try:
    print("Loading model...")
    model = load_model('model.h5')
    
    # السطر ده هيقولنا الموديل متوقع يدخله إيه بالظبط
    print("\n--------------------------------------")
    print(f"✅ Model Input Shape: {model.input_shape}")
    print("--------------------------------------\n")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")