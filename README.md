🌾 Kisaan Bot – Crop Disease Detection System
📌 Overview

Kisaan Bot is a simple and practical AI-based solution designed to help farmers quickly identify crop diseases using just a photo of a leaf.

By sending an image through WhatsApp, users can instantly receive:

Disease name
Confidence level
Suggested medicines
Remedies and precautions

The goal of this project is to make crop disease detection accessible, fast, and easy to use, especially for farmers who may not have direct access to agricultural experts.

🚀 What This Project Does
📸 Detects crop diseases from leaf images
🤖 Uses a trained deep learning model (MobileNetV2)
📱 Works through WhatsApp for easy accessibility
🌍 Supports multiple languages
💊 Provides actionable treatment suggestions
🌱 Supported Crops

Currently supports diseases related to:

Tomato
Potato
Rice
Corn (Maize)
and more classes from the dataset
🌐 Language Support

To make the system usable for more people, it supports:

English
Tamil
Hindi
🧠 How It Works
User sends a leaf image via WhatsApp
The image is processed and resized
The trained model predicts the disease
The system sends back:
Disease name
Confidence score
Medicine
Remedy steps
Preventive measures
🏗️ Project Structure
crop-disease-detection/
│
├── dataset/
│   └── plantvillage dataset/color/
│
├── model/
│   ├── crop_model.h5
│   └── class_names.json
│
├── train_model.py
├── whatsapp_bot.py
├── README.md
└── requirements.txt
⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/your-username/crop-disease-detection.git
cd crop-disease-detection
2. Install Dependencies
pip install -r requirements.txt
🏋️ Training the Model

Run the training script:

python train_model.py

This will:

Train the model using the dataset
Save the trained model in the model/ folder
Store class labels in JSON format
🤖 Running the Bot

Start the Flask server:

python whatsapp_bot.py
📱 WhatsApp Integration (Twilio)

To connect the bot with WhatsApp:

Create a Twilio account
Get your credentials:
Account SID
Auth Token
Update them in the code:
TWILIO_ACCOUNT_SID = "YOUR_TWILIO_SID"
TWILIO_AUTH_TOKEN  = "YOUR_TWILIO_AUTH_TOKEN"
Set your webhook URL to:
http://your-server-url/whatsapp
📊 Sample Response
🔴 Tomato Early Blight
📊 Confidence: 92.5%

💊 Medicine:
Mancozeb (2.5g/L)

🌿 Remedy:
• Remove infected leaves
• Spray fungicide
• Avoid wet leaves

🛡️ Precautions:
• Crop rotation
• Use resistant varieties
⚠️ Important Tips

For better accuracy:

Send only leaf images
Ensure good lighting
Avoid blurry images
Capture a single leaf clearly
🛠️ Technologies Used
Python
TensorFlow / Keras
Flask
Twilio API
NumPy & PIL
