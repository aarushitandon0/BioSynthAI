# BioSynthAI

BioSynthAI is an AI-powered web app that analyzes skin images to detect common skin issues using a custom-trained YOLOv5 model.

## Features

- Upload and preview a selfie image
  
- Select your location from a predefined list
  
- Select your skin type (Oily, Dry, Normal, Combination)
  
- Send data to a backend for skin issue detection and analysis
  
- View detailed analysis results including detected skin issues, weather info, and recommended products
  
- Interactive and modern UI with responsive design

## Tech Stack

- Frontend: React, Axios, CSS
  
- Backend: Flask, PyTorch, OpenCV, YOLOv5
  
- Dataset: Public skin condition datasets (From roboflow)
  
- External APIs: OpenWeatherMap for weather and air quality

## ML Model Training

### Dataset

- The YOLOv5 model was trained on labeled skin condition images from public datasets such as [(https://universe.roboflow.com/nguyen-huyen-cvq6e/skin-problems-detection-jp4jv)]
  
- The dataset includes annotated bounding boxes and class labels for issues like Dark Circle, Melasma, PIH, Blackhead, Cyst, Freckles, Nodule, Papule, Pustule, Skin Pore, Whitehead, and Wrinkle.

## Training
-The YOLOv5 model was trained on a labeled skin condition dataset (e.g., from Roboflow) in Google Colab.

-Training was done for 20 epochs using the Colab GPU environment for faster processing.
```bash
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt --name biosynthai-skin-model
```

-The dataset contained annotated images of common skin issues like Dark Circle, Melasma, Blackhead, Freckles, etc.

-After training, the best weights (best.pt) were saved and downloaded for integration into the backend.

## Model Integration
-The trained model weights (best.pt) are loaded in the Flask backend using YOLOv5’s DetectMultiBackend.

-The backend accepts image uploads, preprocesses images, and runs inference to detect skin issues.

-The highest-confidence detection is selected and returned with additional inf

## Backend API
-Built with Flask and exposes a /analyze POST endpoint.

-Accepts multipart form data: image file, location, and skin type.

-Calls OpenWeatherMap API for weather and air quality data.

-Returns JSON response with detected skin issue, weather info, detailed description, and product recommendations.

## Frontend Setup
Prerequisites
-Node.js and npm/yarn installed
-Installation & Running
```bash
git clone https://github.com/yourusername/biosynthai.git
cd biosynthai
npm install
npm start
```
-Open http://localhost:3000 to use the app.

## Folder Structure
```bash
/backend
  ├─ app.py               # Flask backend with YOLOv5 integration
  ├─ best.pt              # Trained YOLOv5 model weights
  ├─ ml_model.ipnyb  
  └─ yolov5               # Backend dependencies

/src
  ├─ App.js               # React frontend main component
  ├─ App.css              # Styling
  └─ index.js             # React entry point

```

## Future Improvements
-Add more skin issues and improve dataset diversity

-Enhance UI/UX with animations and better mobile support

-Add user authentication and history tracking of analyses

-Expand product recommendations with e-commerce integration





