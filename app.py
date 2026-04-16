import os
import io
import numpy as np
import traceback 
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# --- 1. تحميل الموديل ---
MODEL_PATH = 'model.h5' 

# Automatically reconstruct the model if it was split
if not os.path.exists(MODEL_PATH):
    print("Reassembling model from parts...")
    with open(MODEL_PATH, "wb") as f_out:
        for part in ["model_part1.h5", "model_part2.h5"]:
            if os.path.exists(part):
                with open(part, "rb") as f_in:
                    f_out.write(f_in.read())
            else:
                print(f"Warning: Missing part {part}")

print("Loading model...")
try:
    # Use compile=False to save RAM on free tiers since we solely need inference
    model = load_model(MODEL_PATH, compile=False)
    # طباعة شكل الدخل المتوقع للتأكيد
    print(f"✅ Model loaded! Expected Input Shape: {model.input_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- 2. القائمة الكاملة (38 كلاس) ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # ✅ ضبطنا المقاس على 150 زي ما انت قولت
    img = img.resize((150, 150)) 
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 # Normalization
    return img_array

@app.route('/', methods=['GET'])
def home():
    return "AgroScan AI Server is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        print(f"📷 Received image: {file.filename}") 
        img_bytes = file.read()
        processed_img = prepare_image(img_bytes)

        # التوقع
        prediction = model.predict(processed_img)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        
        # حماية من الأخطاء لو الرقم غريب
        if class_index >= len(CLASS_NAMES):
            raise ValueError(f"Predicted index {class_index} is out of range!")

        full_label = CLASS_NAMES[class_index]
        print(f"🔍 Prediction: {full_label} ({confidence:.2f}%)")

        # معالجة الاسم (فصل النبات عن المرض)
        plant_name = "Unknown"
        disease_name = full_label
        is_healthy = False
        
        if '___' in full_label:
            parts = full_label.split('___')
            plant_name = parts[0].replace('_', ' ').replace('(', '').replace(')', '')
            disease_raw = parts[1].replace('_', ' ')
            is_healthy = 'healthy' in disease_raw.lower()
            disease_name = "Sleem (سليم)" if is_healthy else disease_raw
        else:
            plant_name = full_label
            is_healthy = 'healthy' in full_label.lower()

        treatment_msg = "الري بانتظام ومراقبة النبات." if is_healthy else "يرجى عزل النبات واستشارة مختص."

        return jsonify({
            'plant_name': plant_name,
            'disease_name': disease_name,
            'confidence': f"{confidence:.1f}",
            'is_healthy': is_healthy,
            'treatment': treatment_msg
        })

    except Exception as e:
        print("\n!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!")
        traceback.print_exc() 
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)