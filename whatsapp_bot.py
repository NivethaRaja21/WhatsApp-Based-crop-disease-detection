



from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import tensorflow as tf
import numpy as np
import json
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

TWILIO_ACCOUNT_SID = "YOUR_TWILIO_SID"
TWILIO_AUTH_TOKEN  = "YOUR_TWILIO_AUTH_TOKEN"

print("Loading model...")
model = tf.keras.models.load_model("model/crop_model.h5")
with open("model/class_names.json", "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}
print(f"✅ Model loaded — {len(class_names)} classes")

# ══════════════════════════════════════════════════════════
# DISEASE DATABASE — All 41 classes
# Format: disease_key → {en, ta, hi} → {name, medicine, remedy, precaution}
# ══════════════════════════════════════════════════════════
DISEASE_DB = {

    # ── RICE (Indian crop) ────────────────────────────────
    "Rice___Bacterial_leaf_blight": {
        "en": {"name": "Rice Bacterial Leaf Blight",
               "medicine": "Copper Oxychloride (3g/L) or Streptocycline (0.5g/L)",
               "remedy": "1. Spray Streptocycline + Copper Oxychloride mixture.\n2. Drain fields immediately — avoid flooding.\n3. Remove and burn infected plants.\n4. Avoid excessive nitrogen fertilizer.",
               "precaution": "1. Use resistant varieties like IR64, Swarna.\n2. Treat seeds with Streptocycline before sowing.\n3. Avoid waterlogging in field.\n4. Rotate crops with non-rice crops."},
        "ta": {"name": "நெல் பாக்டீரியா இலை கருகல் நோய்",
               "medicine": "காப்பர் ஆக்சிகுளோரைடு (3கி/லி) அல்லது ஸ்ட்ரெப்டோசைக்ளின் (0.5கி/லி)",
               "remedy": "1. ஸ்ட்ரெப்டோசைக்ளின் + காப்பர் ஆக்சிகுளோரைடு கலந்து தெளிக்கவும்.\n2. வயலில் தண்ணீர் தேங்காமல் உடனே வடிக்கவும்.\n3. பாதிக்கப்பட்ட செடிகளை அகற்றி எரிக்கவும்.\n4. அதிக நைட்ரஜன் உரம் தவிர்க்கவும்.",
               "precaution": "1. IR64, ஸ்வர்ணா போன்ற நோய் எதிர்ப்பு ரகங்களை பயன்படுத்தவும்.\n2. விதைகளை விதைப்பதற்கு முன் ஸ்ட்ரெப்டோசைக்ளினில் ஊறவைக்கவும்.\n3. வயலில் தண்ணீர் தேங்காமல் பார்த்துக்கொள்ளவும்.\n4. நெல் அல்லாத பயிர்களுடன் பயிர் மாற்றம் செய்யவும்."},
        "hi": {"name": "धान बैक्टीरियल पत्ती झुलसा",
               "medicine": "कॉपर ऑक्सीक्लोराइड (3ग्रा/लि) या स्ट्रेप्टोसाइक्लिन (0.5ग्रा/लि)",
               "remedy": "1. स्ट्रेप्टोसाइक्लिन + कॉपर ऑक्सीक्लोराइड मिलाकर छिड़कें।\n2. खेत से पानी तुरंत निकालें।\n3. संक्रमित पौधे जलाकर नष्ट करें।\n4. अधिक नाइट्रोजन उर्वरक से बचें।",
               "precaution": "1. IR64, स्वर्णा जैसी प्रतिरोधी किस्में लगाएं।\n2. बुवाई से पहले बीज को स्ट्रेप्टोसाइक्लिन में भिगोएं।\n3. खेत में जलजमाव न होने दें।\n4. धान के अलावा अन्य फसलों से चक्र अपनाएं।"}
    },

    "Rice___Brown_spot": {
        "en": {"name": "Rice Brown Spot",
               "medicine": "Mancozeb (2.5g/L) or Tricyclazole (0.6g/L)",
               "remedy": "1. Spray Mancozeb or Edifenphos fungicide.\n2. Apply potassium and phosphorus fertilizer.\n3. Drain stagnant water from field.\n4. Remove severely infected tillers.",
               "precaution": "1. Use healthy certified seeds.\n2. Treat seeds with Thiram before sowing.\n3. Maintain balanced soil nutrition — avoid nitrogen excess.\n4. Monitor crop every week."},
        "ta": {"name": "நெல் பழுப்பு புள்ளி நோய்",
               "medicine": "மேன்கோசெப் (2.5கி/லி) அல்லது டிரைசைக்லஸோல் (0.6கி/லி)",
               "remedy": "1. மேன்கோசெப் அல்லது எடிஃபென்ஃபோஸ் தெளிக்கவும்.\n2. பொட்டாசியம் மற்றும் பாஸ்பரஸ் உரம் இடவும்.\n3. வயலில் தேங்கிய தண்ணீரை வடிக்கவும்.\n4. கடுமையாக பாதிக்கப்பட்ட கதிர்களை அகற்றவும்.",
               "precaution": "1. ஆரோக்கியமான சான்றளிக்கப்பட்ட விதைகளை பயன்படுத்தவும்.\n2. விதைப்பதற்கு முன் திரம் கொண்டு விதை நேர்த்தி செய்யவும்.\n3. சமச்சீரான மண் ஊட்டச்சத்தை பராமரிக்கவும்.\n4. வாரம் ஒருமுறை பயிரை கண்காணிக்கவும்."},
        "hi": {"name": "धान भूरा धब्बा रोग",
               "medicine": "मैनकोज़ेब (2.5ग्रा/लि) या ट्राइसाइक्लाज़ोल (0.6ग्रा/लि)",
               "remedy": "1. मैनकोज़ेब या एडिफेनफॉस फफूंदनाशक छिड़कें।\n2. पोटेशियम और फास्फोरस उर्वरक दें।\n3. खेत से रुका पानी निकालें।\n4. गंभीर रूप से संक्रमित पत्तियां हटाएं।",
               "precaution": "1. स्वस्थ प्रमाणित बीजों का उपयोग करें।\n2. बुवाई से पहले थीरम से बीज उपचार करें।\n3. मिट्टी में संतुलित पोषण बनाए रखें।\n4. हर हफ्ते फसल की निगरानी करें।"}
    },

    "Rice___Leaf_smut": {
        "en": {"name": "Rice Leaf Smut",
               "medicine": "Propiconazole (1ml/L) or Carbendazim (1g/L)",
               "remedy": "1. Spray Propiconazole fungicide at booting stage.\n2. Remove and destroy infected plant parts.\n3. Avoid excess nitrogen fertilizer.\n4. Drain excess water from field.",
               "precaution": "1. Use disease-free certified seeds.\n2. Treat seeds with Carbendazim before sowing.\n3. Use resistant rice varieties.\n4. Avoid late planting."},
        "ta": {"name": "நெல் இலை கரி நோய்",
               "medicine": "புரோபிகோனஸோல் (1மி.லி/லி) அல்லது கார்பென்டஸிம் (1கி/லி)",
               "remedy": "1. புட்டிங் நிலையில் புரோபிகோனஸோல் தெளிக்கவும்.\n2. பாதிக்கப்பட்ட பகுதிகளை அகற்றி அழிக்கவும்.\n3. அதிக நைட்ரஜன் உரம் தவிர்க்கவும்.\n4. வயலில் உள்ள தேவையற்ற தண்ணீரை வடிக்கவும்.",
               "precaution": "1. நோயற்ற சான்றளிக்கப்பட்ட விதைகளை பயன்படுத்தவும்.\n2. விதைப்பதற்கு முன் கார்பென்டஸிம் கொண்டு விதை நேர்த்தி செய்யவும்.\n3. நோய் எதிர்ப்பு நெல் ரகங்களை பயன்படுத்தவும்.\n4. தாமதமான நடவு தவிர்க்கவும்."},
        "hi": {"name": "धान पत्ती कंड रोग",
               "medicine": "प्रोपिकोनाज़ोल (1मि.ली/लि) या कार्बेंडाज़िम (1ग्रा/लि)",
               "remedy": "1. बूटिंग चरण में प्रोपिकोनाज़ोल छिड़कें।\n2. संक्रमित भाग हटाकर नष्ट करें।\n3. अधिक नाइट्रोजन से बचें।\n4. खेत से अतिरिक्त पानी निकालें।",
               "precaution": "1. रोग-मुक्त प्रमाणित बीज उपयोग करें।\n2. बुवाई से पहले कार्बेंडाज़िम से बीज उपचार करें।\n3. प्रतिरोधी धान किस्में उपयोग करें।\n4. देर से रोपण न करें।"}
    },

    # ── TOMATO ───────────────────────────────────────────
    "Tomato___Late_blight": {
        "en": {"name": "Tomato Late Blight",
               "medicine": "Copper Oxychloride (3g/L) or Metalaxyl-M (2g/L)",
               "remedy": "1. Remove and destroy infected plants immediately.\n2. Spray Copper Oxychloride or Metalaxyl-M fungicide.\n3. Water only at base — never on leaves.\n4. Improve air circulation between plants.",
               "precaution": "1. Use certified disease-free seeds.\n2. Never plant tomatoes near potatoes.\n3. Rotate crops every season.\n4. Spray preventive fungicide before rainy season."},
        "ta": {"name": "தக்காளி தாமத காய்ச்சல் நோய்",
               "medicine": "காப்பர் ஆக்சிகுளோரைடு (3கி/லி) அல்லது மெட்டலாக்ஸில்-M (2கி/லி)",
               "remedy": "1. பாதிக்கப்பட்ட செடிகளை உடனடியாக அகற்றி அழிக்கவும்.\n2. காப்பர் ஆக்சிகுளோரைடு அல்லது மெட்டலாக்ஸில்-M தெளிக்கவும்.\n3. வேரில் மட்டும் தண்ணீர் ஊற்றவும் — இலையில் படாதீர்கள்.\n4. செடிகளுக்கு இடையே காற்றோட்டம் அதிகரிக்கவும்.",
               "precaution": "1. சான்றளிக்கப்பட்ட விதைகளை மட்டும் பயன்படுத்தவும்.\n2. உருளைக்கிழங்கு அருகில் தக்காளி நடாதீர்கள்.\n3. ஒவ்வொரு பருவமும் பயிர் மாற்றம் செய்யவும்.\n4. மழைக்காலத்திற்கு முன் தடுப்பு மருந்து தெளிக்கவும்."},
        "hi": {"name": "टमाटर लेट ब्लाइट रोग",
               "medicine": "कॉपर ऑक्सीक्लोराइड (3ग्रा/लि) या मेटालैक्सिल-M (2ग्रा/लि)",
               "remedy": "1. संक्रमित पौधे तुरंत हटाएं और नष्ट करें।\n2. कॉपर ऑक्सीक्लोराइड या मेटालैक्सिल-M छिड़कें।\n3. जड़ में पानी दें — पत्तियों पर नहीं।\n4. पौधों के बीच हवा का संचार बढ़ाएं।",
               "precaution": "1. प्रमाणित रोग-मुक्त बीज उपयोग करें।\n2. आलू के पास टमाटर न लगाएं।\n3. हर मौसम में फसल चक्र अपनाएं।\n4. बारिश से पहले रोकथाम फफूंदनाशक छिड़कें।"}
    },

    "Tomato___Early_blight": {
        "en": {"name": "Tomato Early Blight",
               "medicine": "Chlorothalonil (2g/L) or Mancozeb (2.5g/L)",
               "remedy": "1. Remove lower infected leaves immediately.\n2. Spray Chlorothalonil or Mancozeb fungicide.\n3. Water at soil level only — keep leaves dry.\n4. Mulch around base to prevent soil splash.",
               "precaution": "1. Use resistant tomato varieties.\n2. Rotate crops every season.\n3. Remove dead leaves regularly.\n4. Spray neem oil (5ml/L) weekly as prevention."},
        "ta": {"name": "தக்காளி ஆரம்பகால இலை கருகல்",
               "medicine": "குளோரோதலோனில் (2கி/லி) அல்லது மேன்கோசெப் (2.5கி/லி)",
               "remedy": "1. கீழே உள்ள பாதிக்கப்பட்ட இலைகளை உடனே அகற்றவும்.\n2. குளோரோதலோனில் அல்லது மேன்கோசெப் தெளிக்கவும்.\n3. மண்ணில் மட்டும் தண்ணீர் ஊற்றவும்.\n4. வேரின் அடியில் மல்ச் போடுங்கள்.",
               "precaution": "1. நோய் எதிர்ப்பு திறன் கொண்ட ரகங்களை பயன்படுத்தவும்.\n2. ஒவ்வொரு பருவமும் பயிர் மாற்றம் செய்யவும்.\n3. இறந்த இலைகளை தொடர்ந்து சுத்தம் செய்யவும்.\n4. வாரம் ஒருமுறை வேப்பெண்ணெய் (5மி.லி/லி) தெளிக்கவும்."},
        "hi": {"name": "टमाटर अर्ली ब्लाइट रोग",
               "medicine": "क्लोरोथेलोनिल (2ग्रा/लि) या मैनकोज़ेब (2.5ग्रा/लि)",
               "remedy": "1. नीचे की संक्रमित पत्तियां तुरंत हटाएं।\n2. क्लोरोथेलोनिल या मैनकोज़ेब छिड़कें।\n3. मिट्टी में पानी दें — पत्तियां सूखी रखें।\n4. मल्च लगाएं।",
               "precaution": "1. प्रतिरोधी टमाटर किस्में उपयोग करें।\n2. हर मौसम फसल चक्र अपनाएं।\n3. मरी पत्तियां नियमित हटाएं।\n4. हर हफ्ते नीम तेल (5मि.ली/लि) छिड़कें।"}
    },

    "Tomato___Leaf_Mold": {
        "en": {"name": "Tomato Leaf Mold",
               "medicine": "Chlorothalonil (2g/L) or Copper Hydroxide (3g/L)",
               "remedy": "1. Prune crowded branches to improve air flow.\n2. Spray Chlorothalonil or Copper Hydroxide.\n3. Reduce humidity — avoid overhead watering.\n4. Remove and destroy all infected leaves.",
               "precaution": "1. Grow in open areas with good sunlight.\n2. Space plants at least 2 feet apart.\n3. Never wet leaves during watering.\n4. Spray neem oil every 10 days."},
        "ta": {"name": "தக்காளி இலை அச்சு நோய்",
               "medicine": "குளோரோதலோனில் (2கி/லி) அல்லது காப்பர் ஹைட்ராக்சைடு (3கி/லி)",
               "remedy": "1. அடர்ந்த கிளைகளை கத்தரித்து காற்றோட்டம் அதிகரிக்கவும்.\n2. குளோரோதலோனில் அல்லது காப்பர் ஹைட்ராக்சைடு தெளிக்கவும்.\n3. ஈரப்பதத்தை குறைக்கவும்.\n4. பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்.",
               "precaution": "1. நல்ல சூரிய ஒளி வரும் திறந்த வெளியில் வளர்க்கவும்.\n2. செடிகளுக்கிடையே குறைந்தது 2 அடி இடைவெளி விடுங்கள்.\n3. நீர் பாய்ச்சும்போது இலைகளை நனைக்காதீர்கள்.\n4. 10 நாட்களுக்கு ஒருமுறை வேப்பெண்ணெய் தெளிக்கவும்."},
        "hi": {"name": "टमाटर पत्ती फफूंद रोग",
               "medicine": "क्लोरोथेलोनिल (2ग्रा/लि) या कॉपर हाइड्रॉक्साइड (3ग्रा/लि)",
               "remedy": "1. भीड़ वाली शाखाएं काटें।\n2. क्लोरोथेलोनिल या कॉपर हाइड्रॉक्साइड छिड़कें।\n3. नमी कम करें — ऊपर से पानी न दें।\n4. संक्रमित पत्तियां हटाकर नष्ट करें।",
               "precaution": "1. खुली धूप वाली जगह में उगाएं।\n2. पौधों के बीच 2 फीट दूरी रखें।\n3. पानी देते समय पत्तियां न भिगोएं।\n4. हर 10 दिन नीम तेल छिड़कें।"}
    },

    "Tomato___Bacterial_spot": {
        "en": {"name": "Tomato Bacterial Spot",
               "medicine": "Copper Hydroxide (3g/L) or Streptomycin (0.5g/L)",
               "remedy": "1. Spray Copper Hydroxide + Streptomycin mixture.\n2. Remove heavily infected plants immediately.\n3. Never work in field when plants are wet.\n4. Disinfect all tools with bleach before use.",
               "precaution": "1. Use certified disease-free seeds.\n2. Avoid overhead irrigation.\n3. Rotate crops every 2 years.\n4. Remove crop debris after harvest."},
        "ta": {"name": "தக்காளி பாக்டீரியா புள்ளி நோய்",
               "medicine": "காப்பர் ஹைட்ராக்சைடு (3கி/லி) அல்லது ஸ்ட்ரெப்டோமைசின் (0.5கி/லி)",
               "remedy": "1. காப்பர் ஹைட்ராக்சைடு + ஸ்ட்ரெப்டோமைசின் கலந்து தெளிக்கவும்.\n2. அதிகம் பாதிக்கப்பட்ட செடிகளை உடனே அகற்றவும்.\n3. செடிகள் நனைந்திருக்கும்போது வயலில் வேலை செய்யாதீர்கள்.\n4. கருவிகளை பயன்படுத்தும் முன் கிருமிநாசினியால் சுத்தம் செய்யவும்.",
               "precaution": "1. சான்றளிக்கப்பட்ட விதைகளை மட்டும் பயன்படுத்தவும்.\n2. மேலே இருந்து நீர் பாய்ச்சாதீர்கள்.\n3. 2 ஆண்டுகளுக்கு ஒருமுறை பயிர் மாற்றம் செய்யவும்.\n4. அறுவடைக்கு பிறகு பயிர் எச்சங்களை அகற்றவும்."},
        "hi": {"name": "टमाटर बैक्टीरियल स्पॉट",
               "medicine": "कॉपर हाइड्रॉक्साइड (3ग्रा/लि) या स्ट्रेप्टोमाइसिन (0.5ग्रा/लि)",
               "remedy": "1. कॉपर हाइड्रॉक्साइड + स्ट्रेप्टोमाइसिन मिलाकर छिड़कें।\n2. गंभीर रूप से संक्रमित पौधे तुरंत हटाएं।\n3. पौधे गीले हों तो खेत में काम न करें।\n4. उपयोग से पहले सभी उपकरण कीटाणुरहित करें।",
               "precaution": "1. प्रमाणित रोग-मुक्त बीज उपयोग करें।\n2. ऊपर से सिंचाई न करें।\n3. हर 2 साल फसल चक्र अपनाएं।\n4. कटाई के बाद फसल अवशेष हटाएं।"}
    },

    "Tomato___Septoria_leaf_spot": {
        "en": {"name": "Tomato Septoria Leaf Spot",
               "medicine": "Mancozeb (2.5g/L) or Chlorothalonil (2g/L)",
               "remedy": "1. Remove infected lower leaves immediately.\n2. Spray Mancozeb or Chlorothalonil.\n3. Avoid wetting foliage during irrigation.\n4. Mulch around plants.",
               "precaution": "1. Use resistant varieties.\n2. Rotate crops every 2 seasons.\n3. Stake plants to improve air circulation.\n4. Remove plant debris after harvest."},
        "ta": {"name": "தக்காளி செப்டோரியா இலை புள்ளி",
               "medicine": "மேன்கோசெப் (2.5கி/லி) அல்லது குளோரோதலோனில் (2கி/லி)",
               "remedy": "1. பாதிக்கப்பட்ட கீழ் இலைகளை உடனே அகற்றவும்.\n2. மேன்கோசெப் அல்லது குளோரோதலோனில் தெளிக்கவும்.\n3. நீர்ப்பாசனத்தின்போது இலைகளை நனைக்காதீர்கள்.\n4. செடிகளை தூக்கி கட்டுங்கள்.",
               "precaution": "1. எதிர்ப்பு திறன் கொண்ட ரகங்களை பயன்படுத்தவும்.\n2. 2 பருவங்களுக்கு ஒருமுறை பயிர் மாற்றம் செய்யவும்.\n3. காற்றோட்டத்திற்காக செடிகளை தூக்கிக் கட்டவும்.\n4. அறுவடைக்கு பிறகு பயிர் எச்சங்களை அகற்றவும்."},
        "hi": {"name": "टमाटर सेप्टोरिया पत्ती धब्बा",
               "medicine": "मैनकोज़ेब (2.5ग्रा/लि) या क्लोरोथेलोनिल (2ग्रा/लि)",
               "remedy": "1. संक्रमित निचली पत्तियां तुरंत हटाएं।\n2. मैनकोज़ेब या क्लोरोथेलोनिल छिड़कें।\n3. सिंचाई में पत्तियां न भिगोएं।\n4. पौधों के पास मल्च लगाएं।",
               "precaution": "1. प्रतिरोधी किस्में उपयोग करें।\n2. हर 2 मौसम फसल चक्र अपनाएं।\n3. हवा के लिए पौधों को बांधें।\n4. कटाई के बाद अवशेष हटाएं।"}
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "en": {"name": "Tomato Spider Mite Infestation",
               "medicine": "Abamectin (1ml/L) or Spiromesifen (1ml/L)",
               "remedy": "1. Spray Abamectin or Spiromesifen miticide.\n2. Spray water forcefully on underside of leaves.\n3. Remove heavily infested leaves.\n4. Apply neem oil (5ml/L) as organic option.",
               "precaution": "1. Monitor plants weekly for early detection.\n2. Avoid dusty conditions around plants.\n3. Maintain adequate soil moisture.\n4. Avoid excessive nitrogen fertilizer."},
        "ta": {"name": "தக்காளி சிலந்தி பூச்சி தாக்குதல்",
               "medicine": "அபமெக்டின் (1மி.லி/லி) அல்லது ஸ்பைரோமெசிஃபென் (1மி.லி/லி)",
               "remedy": "1. அபமெக்டின் அல்லது ஸ்பைரோமெசிஃபென் தெளிக்கவும்.\n2. இலையின் அடி பகுதியில் அழுத்தமாக தண்ணீர் தெளிக்கவும்.\n3. அதிகமாக பாதிக்கப்பட்ட இலைகளை அகற்றவும்.\n4. கரிம மாற்றாக வேப்பெண்ணெய் (5மி.லி/லி) தெளிக்கவும்.",
               "precaution": "1. வாரம் ஒருமுறை செடிகளை கண்காணிக்கவும்.\n2. செடிகளை தூசி படாமல் பார்த்துக்கொள்ளவும்.\n3. மண்ணில் போதுமான ஈரப்பதம் பராமரிக்கவும்.\n4. அதிக நைட்ரஜன் உரம் தவிர்க்கவும்."},
        "hi": {"name": "टमाटर मकड़ी घुन प्रकोप",
               "medicine": "अबामेक्टिन (1मि.ली/लि) या स्पाइरोमेसिफेन (1मि.ली/लि)",
               "remedy": "1. अबामेक्टिन या स्पाइरोमेसिफेन माइटिसाइड छिड़कें।\n2. पत्तियों के नीचे जोर से पानी छिड़कें।\n3. बुरी तरह प्रभावित पत्तियां हटाएं।\n4. जैविक विकल्प के रूप में नीम तेल (5मि.ली/लि) छिड़कें।",
               "precaution": "1. हर हफ्ते पौधों की निगरानी करें।\n2. पौधों के आसपास धूल से बचाएं।\n3. मिट्टी में पर्याप्त नमी बनाए रखें।\n4. अधिक नाइट्रोजन से बचें।"}
    },

    "Tomato___Target_Spot": {
        "en": {"name": "Tomato Target Spot",
               "medicine": "Azoxystrobin (1ml/L) or Chlorothalonil (2g/L)",
               "remedy": "1. Spray Azoxystrobin or Chlorothalonil fungicide.\n2. Remove infected leaves and destroy them.\n3. Avoid overhead watering.\n4. Ensure proper spacing between plants.",
               "precaution": "1. Use resistant varieties.\n2. Rotate crops every season.\n3. Remove plant debris after harvest.\n4. Apply balanced fertilizer."},
        "ta": {"name": "தக்காளி இலக்கு புள்ளி நோய்",
               "medicine": "அசோக்ஸிஸ்ட்ரோபின் (1மி.லி/லி) அல்லது குளோரோதலோனில் (2கி/லி)",
               "remedy": "1. அசோக்ஸிஸ்ட்ரோபின் அல்லது குளோரோதலோனில் தெளிக்கவும்.\n2. பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்.\n3. மேல்நிலை நீர்ப்பாசனம் தவிர்க்கவும்.\n4. செடிகளுக்கு இடையே சரியான இடைவெளி பராமரிக்கவும்.",
               "precaution": "1. எதிர்ப்பு திறன் கொண்ட ரகங்களை பயன்படுத்தவும்.\n2. ஒவ்வொரு பருவமும் பயிர் மாற்றம் செய்யவும்.\n3. அறுவடைக்கு பிறகு பயிர் எச்சங்களை அகற்றவும்.\n4. சமச்சீரான உரம் இடவும்."},
        "hi": {"name": "टमाटर टार्गेट स्पॉट",
               "medicine": "एज़ोक्सीस्ट्रोबिन (1मि.ली/लि) या क्लोरोथेलोनिल (2ग्रा/लि)",
               "remedy": "1. एज़ोक्सीस्ट्रोबिन या क्लोरोथेलोनिल छिड़कें।\n2. संक्रमित पत्तियां हटाकर नष्ट करें।\n3. ऊपर से पानी न दें।\n4. पौधों के बीच उचित दूरी रखें।",
               "precaution": "1. प्रतिरोधी किस्में उपयोग करें।\n2. हर मौसम फसल चक्र अपनाएं।\n3. कटाई के बाद अवशेष हटाएं।\n4. संतुलित उर्वरक दें।"}
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "en": {"name": "Tomato Yellow Leaf Curl Virus",
               "medicine": "Imidacloprid (0.5ml/L) to kill whitefly vector",
               "remedy": "1. Remove and destroy infected plants immediately.\n2. Spray Imidacloprid to control whiteflies.\n3. Use yellow sticky traps to catch whiteflies.\n4. Cover young plants with insect net.",
               "precaution": "1. Use virus-resistant tomato varieties.\n2. Control whitefly population from early stage.\n3. Remove weeds around field.\n4. Avoid planting near other infected crops."},
        "ta": {"name": "தக்காளி மஞ்சள் இலை சுருள் வைரஸ்",
               "medicine": "இமிடாக்லோபிரிட் (0.5மி.லி/லி) — வெள்ளை ஈக்களை கட்டுப்படுத்த",
               "remedy": "1. பாதிக்கப்பட்ட செடிகளை உடனே அகற்றி அழிக்கவும்.\n2. வெள்ளை ஈக்களை கட்டுப்படுத்த இமிடாக்லோபிரிட் தெளிக்கவும்.\n3. வெள்ளை ஈக்களை பிடிக்க மஞ்சள் பசை பொறி வையுங்கள்.\n4. இளம் செடிகளை பூச்சி வலையால் மூடுங்கள்.",
               "precaution": "1. வைரஸ் எதிர்ப்பு தக்காளி ரகங்களை பயன்படுத்தவும்.\n2. ஆரம்பத்திலேயே வெள்ளை ஈக்களை கட்டுப்படுத்தவும்.\n3. வயலைச் சுற்றி உள்ள களைகளை அகற்றவும்.\n4. பாதிக்கப்பட்ட பிற பயிர்களின் அருகில் நடாதீர்கள்."},
        "hi": {"name": "टमाटर पीला पत्ती मोड़ वायरस",
               "medicine": "इमिडाक्लोप्रिड (0.5मि.ली/लि) — सफेद मक्खी नियंत्रण के लिए",
               "remedy": "1. संक्रमित पौधे तुरंत हटाएं और नष्ट करें।\n2. सफेद मक्खी नियंत्रण के लिए इमिडाक्लोप्रिड छिड़कें।\n3. सफेद मक्खी पकड़ने के लिए पीले चिपचिपे जाल लगाएं।\n4. युवा पौधों को कीट जाल से ढकें।",
               "precaution": "1. वायरस-प्रतिरोधी टमाटर किस्में उपयोग करें।\n2. शुरू से सफेद मक्खी नियंत्रण करें।\n3. खेत के आसपास खरपतवार हटाएं।\n4. अन्य संक्रमित फसलों के पास न लगाएं।"}
    },

    "Tomato___Tomato_mosaic_virus": {
        "en": {"name": "Tomato Mosaic Virus",
               "medicine": "No direct cure — control aphids with Dimethoate (1.5ml/L)",
               "remedy": "1. Remove and destroy infected plants.\n2. Control aphids with Dimethoate spray.\n3. Disinfect hands and tools before touching plants.\n4. Remove weeds that harbor the virus.",
               "precaution": "1. Use virus-resistant seeds.\n2. Control aphid population early.\n3. Avoid using tobacco near plants.\n4. Wash hands before working in field."},
        "ta": {"name": "தக்காளி மொசைக் வைரஸ்",
               "medicine": "நேரடி மருந்து இல்லை — டைமெத்தோயேட் (1.5மி.லி/லி) கொண்டு அசுவினிகளை கட்டுப்படுத்தவும்",
               "remedy": "1. பாதிக்கப்பட்ட செடிகளை அகற்றி அழிக்கவும்.\n2. அசுவினிகளை கட்டுப்படுத்த டைமெத்தோயேட் தெளிக்கவும்.\n3. செடிகளை தொடுவதற்கு முன் கைகளையும் கருவிகளையும் சுத்தம் செய்யவும்.\n4. வைரஸை தாங்கும் களைகளை அகற்றவும்.",
               "precaution": "1. வைரஸ் எதிர்ப்பு விதைகளை பயன்படுத்தவும்.\n2. ஆரம்பத்திலேயே அசுவினிகளை கட்டுப்படுத்தவும்.\n3. செடிகளின் அருகில் புகையிலை பயன்படுத்தாதீர்கள்.\n4. வயலில் வேலை செய்வதற்கு முன் கைகளை கழுவுங்கள்."},
        "hi": {"name": "टमाटर मोज़ेक वायरस",
               "medicine": "सीधा इलाज नहीं — डाइमेथोएट (1.5मि.ली/लि) से एफिड नियंत्रण करें",
               "remedy": "1. संक्रमित पौधे हटाकर नष्ट करें।\n2. एफिड नियंत्रण के लिए डाइमेथोएट छिड़कें।\n3. पौधे छूने से पहले हाथ और उपकरण साफ करें।\n4. वायरस फैलाने वाले खरपतवार हटाएं।",
               "precaution": "1. वायरस-प्रतिरोधी बीज उपयोग करें।\n2. शुरू से एफिड नियंत्रण करें।\n3. पौधों के पास तंबाकू का उपयोग न करें।\n4. खेत में काम से पहले हाथ धोएं।"}
    },

    # ── POTATO ───────────────────────────────────────────
    "Potato___Late_blight": {
        "en": {"name": "Potato Late Blight",
               "medicine": "Metalaxyl-M (2g/L) or Mancozeb (2.5g/L)",
               "remedy": "1. Destroy all infected plants — never compost them.\n2. Spray Metalaxyl-M or Mancozeb immediately.\n3. Ensure proper field drainage — avoid waterlogging.\n4. Hill up soil around potato stems.",
               "precaution": "1. Plant only certified seed potatoes.\n2. Avoid low-lying waterlogged areas.\n3. Spray fungicide before monsoon season.\n4. Harvest before heavy rain season."},
        "ta": {"name": "உருளைக்கிழங்கு தாமத காய்ச்சல்",
               "medicine": "மெட்டலாக்ஸில்-M (2கி/லி) அல்லது மேன்கோசெப் (2.5கி/லி)",
               "remedy": "1. அனைத்து பாதிக்கப்பட்ட செடிகளையும் அழிக்கவும் — உரமாக போடாதீர்கள்.\n2. மெட்டலாக்ஸில்-M அல்லது மேன்கோசெப் உடனே தெளிக்கவும்.\n3. சரியான வடிகால் ஏற்படுத்தவும் — தண்ணீர் தேங்காமல் பார்க்கவும்.\n4. உருளைக்கிழங்கு தண்டு அருகில் மண் குவியுங்கள்.",
               "precaution": "1. சான்றளிக்கப்பட்ட விதை உருளையை மட்டும் நடுங்கள்.\n2. தண்ணீர் தேங்கும் தாழ்வான இடங்களில் நடாதீர்கள்.\n3. மழைக்காலத்திற்கு முன் பூஞ்சைக்கொல்லி தெளிக்கவும்.\n4. கனமழை வருவதற்கு முன் அறுவடை செய்யவும்."},
        "hi": {"name": "आलू लेट ब्लाइट रोग",
               "medicine": "मेटालैक्सिल-M (2ग्रा/लि) या मैनकोज़ेब (2.5ग्रा/लि)",
               "remedy": "1. सभी संक्रमित पौधे नष्ट करें — खाद में न डालें।\n2. मेटालैक्सिल-M या मैनकोज़ेब तुरंत छिड़कें।\n3. जल निकासी सुनिश्चित करें — जलजमाव न हो।\n4. आलू के तनों के पास मिट्टी चढ़ाएं।",
               "precaution": "1. केवल प्रमाणित बीज आलू लगाएं।\n2. नीची, पानी भरने वाली जमीन से बचें।\n3. मानसून से पहले फफूंदनाशक छिड़कें।\n4. भारी बारिश से पहले फसल काटें।"}
    },

    "Potato___Early_blight": {
        "en": {"name": "Potato Early Blight",
               "medicine": "Chlorothalonil (2g/L) or Mancozeb (2.5g/L)",
               "remedy": "1. Spray Chlorothalonil or Mancozeb fungicide.\n2. Remove and destroy infected leaves.\n3. Water at base only — keep leaves dry.\n4. Ensure proper plant spacing for air flow.",
               "precaution": "1. Use resistant potato varieties.\n2. Avoid excess nitrogen fertilizer.\n3. Rotate crops every season.\n4. Apply preventive spray at start of season."},
        "ta": {"name": "உருளைக்கிழங்கு ஆரம்பகால இலை கருகல்",
               "medicine": "குளோரோதலோனில் (2கி/லி) அல்லது மேன்கோசெப் (2.5கி/லி)",
               "remedy": "1. குளோரோதலோனில் அல்லது மேன்கோசெப் தெளிக்கவும்.\n2. பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்.\n3. வேரில் மட்டும் தண்ணீர் ஊற்றவும்.\n4. காற்றோட்டத்திற்காக சரியான இடைவெளியில் நடுங்கள்.",
               "precaution": "1. நோய் எதிர்ப்பு உருளை ரகங்களை பயன்படுத்தவும்.\n2. அதிக நைட்ரஜன் உரம் தவிர்க்கவும்.\n3. ஒவ்வொரு பருவமும் பயிர் மாற்றம் செய்யவும்.\n4. பருவத்தின் ஆரம்பத்தில் தடுப்பு மருந்து தெளிக்கவும்."},
        "hi": {"name": "आलू अर्ली ब्लाइट रोग",
               "medicine": "क्लोरोथेलोनिल (2ग्रा/लि) या मैनकोज़ेब (2.5ग्रा/लि)",
               "remedy": "1. क्लोरोथेलोनिल या मैनकोज़ेब छिड़कें।\n2. संक्रमित पत्तियां हटाकर नष्ट करें।\n3. जड़ में पानी दें — पत्तियां सूखी रखें।\n4. हवा के लिए उचित दूरी रखें।",
               "precaution": "1. प्रतिरोधी आलू किस्में उपयोग करें।\n2. अधिक नाइट्रोजन से बचें।\n3. हर मौसम फसल चक्र अपनाएं।\n4. मौसम की शुरुआत में रोकथाम स्प्रे करें।"}
    },

    # ── CORN ─────────────────────────────────────────────
    "Corn_(maize)___Common_rust_": {
        "en": {"name": "Corn Common Rust",
               "medicine": "Propiconazole (1ml/L) or Tebuconazole (1ml/L)",
               "remedy": "1. Spray Propiconazole or Tebuconazole at first sign.\n2. Apply potassium fertilizer to strengthen plants.\n3. Remove severely infected plants.\n4. Repeat spray every 10-14 days.",
               "precaution": "1. Plant rust-resistant hybrid varieties.\n2. Plant early in season — avoid late planting.\n3. Monitor crop every week.\n4. Maintain proper plant density."},
        "ta": {"name": "மக்காச்சோளம் துரு நோய்",
               "medicine": "புரோபிகோனஸோல் (1மி.லி/லி) அல்லது டெப்யூகோனஸோல் (1மி.லி/லி)",
               "remedy": "1. முதல் அறிகுறி தெரிந்தவுடன் புரோபிகோனஸோல் தெளிக்கவும்.\n2. செடிகளை வலுவாக்க பொட்டாசியம் உரம் இடவும்.\n3. கடுமையாக பாதிக்கப்பட்ட செடிகளை அகற்றவும்.\n4. 10-14 நாட்களுக்கு ஒருமுறை மீண்டும் தெளிக்கவும்.",
               "precaution": "1. துரு எதிர்ப்பு கலப்பின ரகங்களை பயன்படுத்தவும்.\n2. பருவத்தின் ஆரம்பத்தில் நடுங்கள்.\n3. வாரம் ஒருமுறை பயிரை கண்காணிக்கவும்.\n4. சரியான தாவர அடர்த்தியை பராமரிக்கவும்."},
        "hi": {"name": "मक्का सामान्य जंग रोग",
               "medicine": "प्रोपिकोनाज़ोल (1मि.ली/लि) या टेबुकोनाज़ोल (1मि.ली/लि)",
               "remedy": "1. पहले लक्षण पर प्रोपिकोनाज़ोल छिड़कें।\n2. पौधों को मजबूत करने के लिए पोटेशियम उर्वरक दें।\n3. गंभीर रूप से संक्रमित पौधे हटाएं।\n4. हर 10-14 दिन में दोबारा छिड़कें।",
               "precaution": "1. जंग प्रतिरोधी संकर किस्में लगाएं।\n2. मौसम की शुरुआत में लगाएं।\n3. हर हफ्ते निगरानी करें।\n4. उचित पौध घनत्व बनाए रखें।"}
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "en": {"name": "Corn Northern Leaf Blight",
               "medicine": "Azoxystrobin (1ml/L) or Propiconazole (1ml/L)",
               "remedy": "1. Spray Azoxystrobin at tasseling stage for best results.\n2. Remove crop debris after harvest.\n3. Ensure proper field drainage.\n4. Apply balanced NPK fertilizer.",
               "precaution": "1. Use resistant corn varieties.\n2. Rotate corn with soybean or legumes.\n3. Avoid dense planting.\n4. Plow field after harvest to destroy debris."},
        "ta": {"name": "மக்காச்சோளம் வடக்கு இலை கருகல்",
               "medicine": "அசோக்ஸிஸ்ட்ரோபின் (1மி.லி/லி) அல்லது புரோபிகோனஸோல் (1மி.லி/லி)",
               "remedy": "1. பூக்கும் நிலையில் அசோக்ஸிஸ்ட்ரோபின் தெளிப்பது சிறந்த பலன் தரும்.\n2. அறுவடைக்கு பிறகு பயிர் எச்சங்களை அகற்றவும்.\n3. சரியான வடிகால் ஏற்படுத்தவும்.\n4. சமச்சீரான NPK உரம் இடவும்.",
               "precaution": "1. எதிர்ப்பு திறன் கொண்ட மக்காச்சோள ரகங்களை பயன்படுத்தவும்.\n2. சோயாபீன் அல்லது பயிறு வகைகளுடன் பயிர் மாற்றம் செய்யவும்.\n3. அடர்த்தியான நடவு தவிர்க்கவும்.\n4. அறுவடைக்கு பிறகு வயலை உழவு செய்யவும்."},
        "hi": {"name": "मक्का उत्तरी पत्ती झुलसा",
               "medicine": "एज़ोक्सीस्ट्रोबिन (1मि.ली/लि) या प्रोपिकोनाज़ोल (1मि.ली/लि)",
               "remedy": "1. टैसलिंग चरण में एज़ोक्सीस्ट्रोबिन छिड़काव सबसे प्रभावी है।\n2. कटाई के बाद फसल अवशेष हटाएं।\n3. उचित जल निकासी सुनिश्चित करें।\n4. संतुलित NPK उर्वरक दें।",
               "precaution": "1. प्रतिरोधी मक्का किस्में उपयोग करें।\n2. सोयाबीन या फलियों के साथ फसल चक्र अपनाएं।\n3. घनी बुवाई से बचें।\n4. कटाई के बाद खेत जोतें।"}
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "en": {"name": "Corn Gray Leaf Spot",
               "medicine": "Propiconazole (1ml/L) or Azoxystrobin (1ml/L)",
               "remedy": "1. Spray Propiconazole or Azoxystrobin fungicide.\n2. Remove infected lower leaves.\n3. Improve air circulation between plants.\n4. Avoid overhead irrigation.",
               "precaution": "1. Plant resistant corn hybrids.\n2. Rotate with non-grass crops.\n3. Till crop residue after harvest.\n4. Avoid late planting."},
        "ta": {"name": "மக்காச்சோளம் சாம்பல் இலை புள்ளி",
               "medicine": "புரோபிகோனஸோல் (1மி.லி/லி) அல்லது அசோக்ஸிஸ்ட்ரோபின் (1மி.லி/லி)",
               "remedy": "1. புரோபிகோனஸோல் அல்லது அசோக்ஸிஸ்ட்ரோபின் தெளிக்கவும்.\n2. பாதிக்கப்பட்ட கீழ் இலைகளை அகற்றவும்.\n3. செடிகளுக்கு இடையே காற்றோட்டம் அதிகரிக்கவும்.\n4. மேல்நிலை நீர்ப்பாசனம் தவிர்க்கவும்.",
               "precaution": "1. எதிர்ப்பு திறன் கொண்ட கலப்பின ரகங்களை பயன்படுத்தவும்.\n2. புல் அல்லாத பயிர்களுடன் பயிர் மாற்றம் செய்யவும்.\n3. அறுவடைக்கு பிறகு பயிர் எச்சங்களை உழவு செய்யவும்.\n4. தாமதமான நடவு தவிர்க்கவும்."},
        "hi": {"name": "मक्का भूरा पत्ती धब्बा",
               "medicine": "प्रोपिकोनाज़ोल (1मि.ली/लि) या एज़ोक्सीस्ट्रोबिन (1मि.ली/लि)",
               "remedy": "1. प्रोपिकोनाज़ोल या एज़ोक्सीस्ट्रोबिन छिड़कें।\n2. संक्रमित निचली पत्तियां हटाएं।\n3. पौधों के बीच हवा का संचार बढ़ाएं।\n4. ऊपर से सिंचाई से बचें।",
               "precaution": "1. प्रतिरोधी मक्का संकर किस्में लगाएं।\n2. घास रहित फसलों के साथ चक्र अपनाएं।\n3. कटाई के बाद अवशेष जोतें।\n4. देर से रोपण न करें।"}
    },

    # ── HEALTHY classes ───────────────────────────────────
    "healthy": {
        "en": {"name": "Healthy Crop ✅",
               "medicine": "No medicine needed",
               "remedy": "Your crop looks healthy! No disease detected. Keep up the good work!",
               "precaution": "1. Continue regular watering and fertilization.\n2. Monitor crop weekly for early disease signs.\n3. Maintain proper spacing between plants.\n4. Apply neem oil (5ml/L) monthly as prevention."},
        "ta": {"name": "ஆரோக்கியமான பயிர் ✅",
               "medicine": "மருந்து தேவையில்லை",
               "remedy": "உங்கள் பயிர் ஆரோக்கியமாக உள்ளது! எந்த நோயும் கண்டறியப்படவில்லை. தொடர்ந்து இதே முறையில் கவனிக்கவும்!",
               "precaution": "1. தொடர்ந்து சரியான நீர் பாய்ச்சல் மற்றும் உரமிடுதல் செய்யவும்.\n2. நோயின் ஆரம்ப அறிகுறிகளுக்கு வாரம் ஒருமுறை பயிரை கண்காணிக்கவும்.\n3. செடிகளுக்கிடையே சரியான இடைவெளி பராமரிக்கவும்.\n4. தடுப்பு நடவடிக்கையாக மாதம் ஒருமுறை வேப்பெண்ணெய் (5மி.லி/லி) தெளிக்கவும்."},
        "hi": {"name": "स्वस्थ फसल ✅",
               "medicine": "कोई दवा की जरूरत नहीं",
               "remedy": "आपकी फसल स्वस्थ है! कोई बीमारी नहीं मिली। ऐसे ही देखभाल जारी रखें!",
               "precaution": "1. नियमित सिंचाई और उर्वरक जारी रखें।\n2. बीमारी के शुरुआती लक्षणों के लिए साप्ताहिक निगरानी करें।\n3. पौधों के बीच उचित दूरी बनाए रखें।\n4. रोकथाम के लिए मासिक नीम तेल (5मि.ली/लि) छिड़कें।"}
    },
}

def get_disease_info(label):
    # Exact match first
    if label in DISEASE_DB:
        return DISEASE_DB[label], "healthy" in label
    # Partial match
    for key in DISEASE_DB:
        if key.lower() in label.lower() or label.lower() in key.lower():
            return DISEASE_DB[key], "healthy" in key
    if "healthy" in label.lower():
        return DISEASE_DB["healthy"], True
    return None, False

def build_reply(label, confidence, lang):
    info, is_healthy = get_disease_info(label)
    icon = "✅" if is_healthy else "🔴"

    if info is None:
        # Generic fallback with actual disease name
        clean = label.replace("___", " — ").replace("_", " ")
        msgs = {
            "en": (f"⚠️ *{clean}*\n📊 Confidence: {confidence:.1%}\n\n"
                   f"💊 *Remedy:* Consult your nearest agricultural office.\n"
                   f"📞 Kisan Call Centre: 1800-180-1551 (Free)\n\n"
                   f"📸 Send another leaf photo."),
            "ta": (f"⚠️ *{clean}*\n📊 நம்பகத்தன்மை: {confidence:.1%}\n\n"
                   f"💊 *தீர்வு:* உங்கள் அருகிலுள்ள வேளாண்மை அலுவலகத்தை அணுகவும்.\n"
                   f"📞 கிசான் அழைப்பு மையம்: 1800-180-1551 (இலவசம்)\n\n"
                   f"📸 மற்றொரு இலை படம் அனுப்பவும்."),
            "hi": (f"⚠️ *{clean}*\n📊 विश्वास: {confidence:.1%}\n\n"
                   f"💊 *उपाय:* नजदीकी कृषि कार्यालय से संपर्क करें।\n"
                   f"📞 किसान कॉल सेंटर: 1800-180-1551 (निःशुल्क)\n\n"
                   f"📸 दूसरी पत्ती की फोटो भेजें।")
        }
        return msgs.get(lang, msgs["en"])

    l = info[lang]
    if lang == "en":
        return (f"{icon} *{l['name']}*\n"
                f"📊 Confidence: {confidence:.1%}\n\n"
                f"💊 *Medicine:* {l['medicine']}\n\n"
                f"🌿 *Remedy:*\n{l['remedy']}\n\n"
                f"🛡️ *Precautions:*\n{l['precaution']}\n\n"
                f"📞 Kisan helpline: 1800-180-1551\n"
                f"📸 Send another leaf photo anytime.")
    elif lang == "ta":
        return (f"{icon} *{l['name']}*\n"
                f"📊 நம்பகத்தன்மை: {confidence:.1%}\n\n"
                f"💊 *மருந்து:* {l['medicine']}\n\n"
                f"🌿 *தீர்வு:*\n{l['remedy']}\n\n"
                f"🛡️ *முன்னெச்சரிக்கை:*\n{l['precaution']}\n\n"
                f"📞 கிசான் உதவி எண்: 1800-180-1551\n"
                f"📸 மற்றொரு இலை படம் அனுப்பவும்.")
    elif lang == "hi":
        return (f"{icon} *{l['name']}*\n"
                f"📊 विश्वास: {confidence:.1%}\n\n"
                f"💊 *दवा:* {l['medicine']}\n\n"
                f"🌿 *उपाय:*\n{l['remedy']}\n\n"
                f"🛡️ *सावधानियां:*\n{l['precaution']}\n\n"
                f"📞 किसान हेल्पलाइन: 1800-180-1551\n"
                f"📸 दूसरी पत्ती की फोटो भेजें।")

def predict_from_url(image_url):
    response = requests.get(image_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    predictions = model.predict(img_array)
    idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][idx])
    return class_names[idx], confidence

user_lang  = {}

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    from_number  = request.values.get("From", "")
    incoming_msg = request.values.get("Body", "").strip()
    media_url    = request.values.get("MediaUrl0", None)
    num_media    = int(request.values.get("NumMedia", 0))

    resp = MessagingResponse()
    msg  = resp.message()
    lang = user_lang.get(from_number, None)

    # Language selection commands
    if incoming_msg in ["1", "English", "english", "ENGLISH"]:
        user_lang[from_number] = "en"
        msg.body("✅ Language set to English!\n\n📸 Now send a clear photo of your crop *leaf* to detect disease.\n\n⚠️ Important: Send only LEAF photos (not fruit, stem or full plant) for accurate results.")
        return str(resp)
    elif incoming_msg in ["2", "Tamil", "tamil", "தமிழ்"]:
        user_lang[from_number] = "ta"
        msg.body("✅ மொழி தமிழாக அமைக்கப்பட்டது!\n\n📸 இப்போது உங்கள் பயிரின் *இலை* படம் அனுப்பவும்.\n\n⚠️ முக்கியம்: சரியான முடிவுகளுக்கு இலை படங்களை மட்டுமே அனுப்பவும் (பழம், தண்டு அல்ல).")
        return str(resp)
    elif incoming_msg in ["3", "Hindi", "hindi", "हिंदी"]:
        user_lang[from_number] = "hi"
        msg.body("✅ भाषा हिंदी में सेट की गई!\n\n📸 अब अपनी फसल की *पत्ती* की फोटो भेजें।\n\n⚠️ महत्वपूर्ण: सटीक परिणाम के लिए केवल पत्ती की फोटो भेजें (फल या तना नहीं)।")
        return str(resp)

    # No language selected yet
    if lang is None:
        msg.body(
            "🌾 *Kisaan Bot — Crop Disease Detector*\n"
            "🌾 *கிசான் பாட் — பயிர் நோய் கண்டறிதல்*\n"
            "🌾 *किसान बॉट — फसल रोग पहचान*\n\n"
            "Please select your language:\n"
            "மொழியை தேர்ந்தெடுக்கவும்:\n"
            "भाषा चुनें:\n\n"
            "1️⃣  English\n"
            "2️⃣  தமிழ் (Tamil)\n"
            "3️⃣  हिंदी (Hindi)\n\n"
            "Reply with *1*, *2* or *3*"
        )
        return str(resp)

    # Image received
    if num_media > 0 and media_url:
        try:
            label, confidence = predict_from_url(media_url)
            reply = build_reply(label, confidence, lang)
            msg.body(reply)
        except Exception as e:
            err = {
                "en": "❌ Could not analyse image.\nPlease send a clear, well-lit *leaf* photo and try again.",
                "ta": "❌ படத்தை பகுப்பாய்வு செய்ய முடியவில்லை.\nதெளிவான *இலை* படம் அனுப்பி மீண்டும் முயற்சிக்கவும்.",
                "hi": "❌ फोटो का विश्लेषण नहीं हो सका।\nस्पष्ट *पत्ती* फोटो भेजकर दोबारा प्रयास करें।"
            }
            msg.body(err.get(lang, err["en"]))
    else:
        # Text message — show help
        help_msg = {
            "en": ("📸 Please send a clear photo of your crop *leaf*.\n\n"
                   "⚠️ *Tips for best results:*\n"
                   "• Use a flat single leaf\n"
                   "• Good natural lighting\n"
                   "• Only the leaf in the photo\n"
                   "• Not fruit, stem or full plant\n\n"
                   "🌾 Supported crops: Tomato, Potato, Corn, Rice, Apple, Grape & more\n\n"
                   "Type *1* English | *2* Tamil | *3* Hindi"),
            "ta": ("📸 உங்கள் பயிரின் *இலை* படம் அனுப்பவும்.\n\n"
                   "⚠️ *சிறந்த முடிவுகளுக்கு:*\n"
                   "• ஒரு தட்டையான இலை படம் எடுக்கவும்\n"
                   "• நல்ல இயற்கை வெளிச்சம் இருக்கட்டும்\n"
                   "• படத்தில் இலை மட்டும் இருக்கட்டும்\n"
                   "• பழம், தண்டு அல்லது முழு செடி வேண்டாம்\n\n"
                   "🌾 ஆதரிக்கப்படும் பயிர்கள்: தக்காளி, உருளை, நெல், மக்காச்சோளம் மற்றும் பலவற்றின் இலைகள்\n\n"
                   "*1* English | *2* தமிழ் | *3* Hindi என்று தட்டச்சு செய்யவும்"),
            "hi": ("📸 अपनी फसल की *पत्ती* की फोटो भेजें।\n\n"
                   "⚠️ *बेहतर परिणाम के लिए:*\n"
                   "• एक सपाट पत्ती की फोटो लें\n"
                   "• अच्छी प्राकृतिक रोशनी हो\n"
                   "• फोटो में सिर्फ पत्ती हो\n"
                   "• फल, तना या पूरा पौधा नहीं\n\n"
                   "🌾 समर्थित फसलें: टमाटर, आलू, मक्का, धान, सेब और अधिक\n\n"
                   "*1* English | *2* Tamil | *3* हिंदी टाइप करें")
        }
        msg.body(help_msg.get(lang, help_msg["en"]))

    return str(resp)

if __name__ == "__main__":
    print("\n🌾 Kisaan Bot starting on port 5000...")
    print("   Supported crops: Tomato, Potato, Corn, Rice + more")
    print("   Languages: English, Tamil, Hindi\n")
    app.run(debug=False, port=5000)