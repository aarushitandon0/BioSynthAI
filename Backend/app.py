from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import requests
from flask_cors import CORS
import pathlib
import sys
import torch
import cv2
import numpy as np

# Fix PosixPath issue on Windows
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

# Add yolov5 repo to sys.path to import utils and models properly
yolov5_path = os.path.join(os.path.dirname(__file__), r'C:\Users\AARUSHI TANDON\OneDrive\Python\BioSynthAI\Backend\yolov5')
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)

# Now import YOLOv5 utilities and model
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# Flask setup
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

OPENWEATHER_API_KEY = 'bba424279501c51cb00b0b800d1a12dd'

PRODUCT_RECOMMENDATIONS = {
    "Dark Circle": ["Eye Cream with Vitamin K", "Caffeine Serum", "Cold Compress"],
    "Melasma": ["Sunscreen SPF 50+", "Vitamin C Serum", "Hydroquinone Cream"],
    "PIH": ["Niacinamide Serum", "Azelaic Acid", "Retinoid Cream"],
    "blackhead": ["Salicylic Acid Cleanser", "Clay Mask", "Non-comedogenic Moisturizer"],
    "cyst": ["Benzoyl Peroxide", "Antibiotic Cream", "Consult Dermatologist"],
    "freckles": ["Sunscreen SPF 50+", "Vitamin C Serum", "Chemical Peels"],
    "nodule": ["Topical Steroids", "Consult Dermatologist"],
    "papule": ["Retinoids", "Salicylic Acid", "Gentle Cleanser"],
    "pustule": ["Benzoyl Peroxide", "Antibiotic Gel", "Oil-free Moisturizer"],
    "skin-pore": ["Exfoliating Scrub", "Clay Mask", "Non-comedogenic Moisturizer"],
    "whitehead": ["Salicylic Acid Cleanser", "Retinoids", "Oil-free Moisturizer"],
    "wrinkle": ["Retinol Cream", "Hyaluronic Acid Serum", "Sunscreen SPF 50+"]
}
skin_issue_descriptions = {
    "Dark Circle": {
        "cause": "Often caused by lack of sleep, genetics, dehydration, or aging which leads to thinning skin under the eyes.",
        "prevention": "Get enough sleep, stay hydrated, protect skin from sun exposure, and manage allergies.",
        "treatment": "Use eye creams with Vitamin K, caffeine serums, cold compresses, and consult a dermatologist if severe."
    },
    "Melasma": {
        "cause": "Triggered by sun exposure, hormonal changes (like pregnancy), or certain medications.",
        "prevention": "Apply broad-spectrum sunscreen daily, avoid excessive sun exposure, and wear protective clothing.",
        "treatment": "Use topical treatments like hydroquinone, retinoids, and seek dermatological procedures if necessary."
    },
    "PIH": {  
        "cause": "Dark spots left after acne, injury, or inflammation heals.",
        "prevention": "Avoid picking or squeezing skin lesions and protect skin from the sun.",
        "treatment": "Use brightening agents like Vitamin C, niacinamide, and chemical exfoliants."
    },
    "blackhead": {
        "cause": "Clogged hair follicles filled with excess oil and dead skin cells exposed to air.",
        "prevention": "Regular cleansing, exfoliation, and avoiding heavy makeup or oily products.",
        "treatment": "Use salicylic acid, retinoids, and professional extractions if needed."
    },
    "cyst": {
        "cause": "Deep, painful lumps filled with pus caused by infection or clogged pores.",
        "prevention": "Avoid squeezing pimples and keep skin clean.",
        "treatment": "See a dermatologist for possible drainage or corticosteroid injections."
    },
    "freckles": {
        "cause": "Clusters of concentrated melanin, often genetic and worsened by sun exposure.",
        "prevention": "Sun protection and avoiding excessive UV exposure.",
        "treatment": "Usually harmless but can use sunscreen and skin-lightening creams."
    },
    "nodule": {
        "cause": "Large, solid, painful lumps beneath the skin caused by deep acne inflammation.",
        "prevention": "Early acne treatment and avoiding picking.",
        "treatment": "Requires medical treatment, possibly oral antibiotics or isotretinoin."
    },
    "papule": {
        "cause": "Small, raised, red bumps caused by inflammation or clogged pores.",
        "prevention": "Proper skin cleansing and avoiding irritants.",
        "treatment": "Topical treatments like benzoyl peroxide or salicylic acid."
    },
    "pustule": {
        "cause": "Pimples filled with pus caused by bacterial infection of clogged pores.",
        "prevention": "Keep skin clean and avoid squeezing.",
        "treatment": "Use antibacterial creams, benzoyl peroxide, and consult a dermatologist for severe cases."
    },
    "skin-pore": {
        "cause": "Visible openings on the skin surface, can become enlarged due to excess oil or loss of elasticity.",
        "prevention": "Proper cleansing and exfoliation to remove buildup.",
        "treatment": "Use products containing niacinamide and retinoids to tighten pores."
    },
    "whitehead": {
        "cause": "Closed clogged pores filled with oil and dead skin cells.",
        "prevention": "Regular gentle cleansing and exfoliation.",
        "treatment": "Use salicylic acid and topical retinoids."
    },
    "wrinkle": {
        "cause": "Caused by aging, sun exposure, smoking, and loss of skin elasticity.",
        "prevention": "Use sunscreen, moisturize regularly, avoid smoking and excessive sun.",
        "treatment": "Use retinoids, peptides, antioxidants, and consider dermatological procedures."
    }
}

# Initialize YOLOv5 model
device = select_device('cpu')
# Make sure 'best.pt' is in your Backend folder or provide full path here:
model = DetectMultiBackend('best.pt', device=device)
model.eval()
model.conf = 0.3  # confidence threshold

@app.route('/')
def home():
    return "Welcome to BioSynthAI backend!"

def get_weather_data(city_name):
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={OPENWEATHER_API_KEY}&units=metric"
    weather_response = requests.get(weather_url)
    if weather_response.status_code != 200:
        return None
    weather_data = weather_response.json()

    lat = weather_data['coord']['lat']
    lon = weather_data['coord']['lon']
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    pollution_response = requests.get(pollution_url)
    if pollution_response.status_code != 200:
        return None
    pollution_data = pollution_response.json()

    return {
        "temperature_celsius": weather_data['main']['temp'],
        "humidity_percent": weather_data['main']['humidity'],
        "description": weather_data['weather'][0]['description'],
        "aqi": pollution_data['list'][0]['main']['aqi']
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in request"}), 400

    image = request.files['image']
    location = request.form.get('location')
    skin_type = request.form.get('skin_type')

    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400

    if not location or not skin_type:
        return jsonify({"error": "Location and skin_type are required"}), 400

    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    # Load and preprocess image
    img0 = cv2.imread(filepath)  # BGR format
    img = letterbox(img0, new_shape=(640, 640))[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, then to 3xHxW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0  # Normalize to 0-1
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)

    if len(pred[0]) == 0:
        detected_issue = "No issue detected"
    else:
        top = pred[0][pred[0][:, 4].argmax()]  # get detection with highest confidence
        class_id = int(top[5].item())
        detected_issue = model.names[class_id]

    weather_info = get_weather_data(location)
    if not weather_info:
        weather_info = {"error": "Could not fetch weather data for location"}

    recommended_products = PRODUCT_RECOMMENDATIONS.get(detected_issue, [])
    description = skin_issue_descriptions.get(detected_issue, {
    "cause": "Description not available",
    "prevention": "Description not available",
    "treatment": "Description not available"})

    # Clean up uploaded image file
    os.remove(filepath)

    return jsonify({
        "detected_issue": detected_issue,
        "skin_type": skin_type,
        "location": location,
        "weather_info": weather_info,
        "recommended_products": recommended_products,
        "description": description
    })

if __name__ == '__main__':
    app.run(debug=True)
