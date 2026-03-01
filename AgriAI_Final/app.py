"""
=======================================================
  INDIAN CROP YIELD PREDICTION — MULTILINGUAL STREAMLIT APP
  Supports 12 Indian Languages | Regional Crop Recommendations
=======================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Fasal Utpadan / Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  TRANSLATIONS — 12 INDIAN LANGUAGES + ENGLISH
# ─────────────────────────────────────────────
TRANSLATIONS = {
    "English": {
        "app_title": "🌾 Crop Yield Predictor",
        "app_subtitle": "AI-powered agricultural intelligence for Indian farmers",
        "language_label": "🌐 Select Language / भाषा चुनें",
        "tab_predict": "🌱 Predict Yield",
        "tab_recommend": "💡 Crop Recommendation",
        "tab_batch": "📋 Batch Predict",
        "tab_about": "ℹ️ About",
        "select_state": "Select State",
        "select_crop": "Select Crop",
        "select_season": "Select Season",
        "select_year": "Select Year",
        "fertilizer": "Fertilizer (kg/ha)",
        "pesticide": "Pesticide (kg/ha)",
        "soil_header": "🪨 Soil Parameters",
        "nitrogen": "Nitrogen (N) kg/ha",
        "phosphorus": "Phosphorus (P) kg/ha",
        "potassium": "Potassium (K) kg/ha",
        "soil_ph": "Soil pH",
        "weather_header": "🌤️ Weather Parameters",
        "temperature": "Avg Temperature (°C)",
        "rainfall": "Annual Rainfall (mm)",
        "humidity": "Avg Humidity (%)",
        "lag_header": "📅 Historical Data (Optional)",
        "yield_lag1": "Last Year's Yield (kg/ha)",
        "yield_lag2": "2 Years Ago Yield (kg/ha)",
        "predict_btn": "🔮 Predict Yield",
        "predicted_yield": "Predicted Yield",
        "unit": "kg / hectare",
        "model_not_found": "⚠️ Model file not found. Please train the model first by running the main ML script.",
        "result_header": "📊 Prediction Result",
        "good_yield": "✅ Above Average Yield",
        "avg_yield": "⚠️ Average Yield",
        "low_yield": "❌ Below Average Yield",
        "tips_header": "💡 Farming Tips",
        "recommend_header": "🌿 Regional Crop Recommendations",
        "recommend_state": "Select Your State",
        "recommend_season": "Select Season",
        "recommend_btn": "🔍 Get Recommendations",
        "top_crops": "Recommended Crops for Your Region",
        "batch_header": "📋 Batch Prediction",
        "upload_csv": "Upload CSV File",
        "download_results": "⬇️ Download Results",
        "about_header": "About This App",
        "model_stats": "🤖 Model Performance",
        "crop_input_header": "🌾 Crop & Season",
        "agri_input_header": "🧪 Fertilizer & Pesticide",
        "loading": "Computing yield prediction...",
        "report_title": "Crop Yield Prediction Report",
        "report_download": "📥 Download Report (PDF)",
        "chart_soil": "Soil Nutrient Distribution",
        "chart_trend": "Yield Trend Analysis",
        "chart_weather": "Weather Overview",
        "about_dev": "Developed by",
        "about_model_title": "Machine Learning Model",
        "about_features_title": "Features Used",
        "about_languages_title": "Languages Supported",
        "about_files_title": "Required Files",
    },
    "हिंदी (Hindi)": {
        "app_title": "🌾 फसल उत्पादन भविष्यवाणी",
        "app_subtitle": "भारतीय किसानों के लिए AI-संचालित कृषि बुद्धिमत्ता",
        "language_label": "🌐 भाषा चुनें",
        "tab_predict": "🌱 उत्पादन भविष्यवाणी",
        "tab_recommend": "💡 फसल सिफारिश",
        "tab_batch": "📋 बैच भविष्यवाणी",
        "tab_about": "ℹ️ जानकारी",
        "select_state": "राज्य चुनें",
        "select_crop": "फसल चुनें",
        "select_season": "मौसम चुनें",
        "select_year": "वर्ष चुनें",
        "fertilizer": "उर्वरक (kg/ha)",
        "pesticide": "कीटनाशक (kg/ha)",
        "soil_header": "🪨 मिट्टी के पैरामीटर",
        "nitrogen": "नाइट्रोजन (N) kg/ha",
        "phosphorus": "फास्फोरस (P) kg/ha",
        "potassium": "पोटेशियम (K) kg/ha",
        "soil_ph": "मिट्टी का pH",
        "weather_header": "🌤️ मौसम पैरामीटर",
        "temperature": "औसत तापमान (°C)",
        "rainfall": "वार्षिक वर्षा (mm)",
        "humidity": "औसत आर्द्रता (%)",
        "lag_header": "📅 ऐतिहासिक डेटा (वैकल्पिक)",
        "yield_lag1": "पिछले साल की उपज (kg/ha)",
        "yield_lag2": "2 साल पहले की उपज (kg/ha)",
        "predict_btn": "🔮 उत्पादन की भविष्यवाणी करें",
        "predicted_yield": "अनुमानित उत्पादन",
        "unit": "kg / हेक्टेयर",
        "model_not_found": "⚠️ मॉडल फ़ाइल नहीं मिली। कृपया पहले ML स्क्रिप्ट चलाएं।",
        "result_header": "📊 भविष्यवाणी परिणाम",
        "good_yield": "✅ औसत से अधिक उपज",
        "avg_yield": "⚠️ औसत उपज",
        "low_yield": "❌ औसत से कम उपज",
        "tips_header": "💡 कृषि सुझाव",
        "recommend_header": "🌿 क्षेत्रीय फसल सिफारिशें",
        "recommend_state": "अपना राज्य चुनें",
        "recommend_season": "मौसम चुनें",
        "recommend_btn": "🔍 सिफारिशें प्राप्त करें",
        "top_crops": "आपके क्षेत्र के लिए अनुशंसित फसलें",
        "batch_header": "📋 बैच भविष्यवाणी",
        "upload_csv": "CSV फ़ाइल अपलोड करें",
        "download_results": "⬇️ परिणाम डाउनलोड करें",
        "about_header": "इस ऐप के बारे में",
        "model_stats": "🤖 मॉडल प्रदर्शन",
        "crop_input_header": "🌾 फसल और मौसम",
        "agri_input_header": "🧪 उर्वरक और कीटनाशक",
        "loading": "उपज भविष्यवाणी की गणना हो रही है...",
        "report_title": "फसल उत्पादन भविष्यवाणी रिपोर्ट",
        "report_download": "📥 रिपोर्ट डाउनलोड करें (PDF)",
        "chart_soil": "मिट्टी पोषक वितरण",
        "chart_trend": "उपज प्रवृत्ति विश्लेषण",
        "chart_weather": "मौसम अवलोकन",
        "about_dev": "द्वारा विकसित",
        "about_model_title": "मशीन लर्निंग मॉडल",
        "about_features_title": "उपयोग की गई विशेषताएं",
        "about_languages_title": "समर्थित भाषाएं",
        "about_files_title": "आवश्यक फ़ाइलें",
    },
    "বাংলা (Bengali)": {
        "app_title": "🌾 ফসল উৎপাদন পূর্বাভাস",
        "app_subtitle": "ভারতীয় কৃষকদের জন্য AI-চালিত কৃষি বুদ্ধিমত্তা",
        "language_label": "🌐 ভাষা নির্বাচন করুন",
        "tab_predict": "🌱 ফলন পূর্বাভাস",
        "tab_recommend": "💡 ফসলের সুপারিশ",
        "tab_batch": "📋 ব্যাচ পূর্বাভাস",
        "tab_about": "ℹ️ সম্পর্কে",
        "select_state": "রাজ্য নির্বাচন করুন",
        "select_crop": "ফসল নির্বাচন করুন",
        "select_season": "ঋতু নির্বাচন করুন",
        "select_year": "বছর নির্বাচন করুন",
        "fertilizer": "সার (kg/ha)",
        "pesticide": "কীটনাশক (kg/ha)",
        "soil_header": "🪨 মাটির পরামিতি",
        "nitrogen": "নাইট্রোজেন (N) kg/ha",
        "phosphorus": "ফসফরাস (P) kg/ha",
        "potassium": "পটাসিয়াম (K) kg/ha",
        "soil_ph": "মাটির pH",
        "weather_header": "🌤️ আবহাওয়ার পরামিতি",
        "temperature": "গড় তাপমাত্রা (°C)",
        "rainfall": "বার্ষিক বৃষ্টিপাত (mm)",
        "humidity": "গড় আর্দ্রতা (%)",
        "lag_header": "📅 ঐতিহাসিক তথ্য (ঐচ্ছিক)",
        "yield_lag1": "গত বছরের ফলন (kg/ha)",
        "yield_lag2": "২ বছর আগের ফলন (kg/ha)",
        "predict_btn": "🔮 ফলন পূর্বাভাস দিন",
        "predicted_yield": "পূর্বাভাসিত ফলন",
        "unit": "kg / হেক্টর",
        "model_not_found": "⚠️ মডেল ফাইল পাওয়া যায়নি। ML স্ক্রিপ্ট চালান।",
        "result_header": "📊 পূর্বাভাসের ফলাফল",
        "good_yield": "✅ গড়ের উপরে ফলন",
        "avg_yield": "⚠️ গড় ফলন",
        "low_yield": "❌ গড়ের নিচে ফলন",
        "tips_header": "💡 কৃষি পরামর্শ",
        "recommend_header": "🌿 আঞ্চলিক ফসলের সুপারিশ",
        "recommend_state": "আপনার রাজ্য নির্বাচন করুন",
        "recommend_season": "ঋতু নির্বাচন করুন",
        "recommend_btn": "🔍 সুপারিশ পান",
        "top_crops": "আপনার অঞ্চলের জন্য প্রস্তাবিত ফসল",
        "batch_header": "📋 ব্যাচ পূর্বাভাস",
        "upload_csv": "CSV ফাইল আপলোড করুন",
        "download_results": "⬇️ ফলাফল ডাউনলোড করুন",
        "about_header": "এই অ্যাপ সম্পর্কে",
        "model_stats": "🤖 মডেলের কার্যক্ষমতা",
        "crop_input_header": "🌾 ফসল ও ঋতু",
        "agri_input_header": "🧪 সার ও কীটনাশক",
        "loading": "ফলন পূর্বাভাস গণনা করা হচ্ছে...",
        "report_title": "ফসল উৎপাদন পূর্বাভাস রিপোর্ট",
        "report_download": "📥 রিপোর্ট ডাউনলোড করুন (PDF)",
        "chart_soil": "মাটির পুষ্টি বিতরণ",
        "chart_trend": "ফলন প্রবণতা বিশ্লেষণ",
        "chart_weather": "আবহাওয়া সংক্ষিপ্তসার",
        "about_dev": "তৈরি করেছেন",
        "about_model_title": "মেশিন লার্নিং মডেল",
        "about_features_title": "ব্যবহৃত বৈশিষ্ট্য",
        "about_languages_title": "সমর্থিত ভাষা",
        "about_files_title": "প্রয়োজনীয় ফাইল",
    },
    "தமிழ் (Tamil)": {
        "app_title": "🌾 பயிர் மகசூல் கணிப்பு",
        "app_subtitle": "இந்திய விவசாயிகளுக்கான AI-இயங்கும் வேளாண் அறிவு",
        "language_label": "🌐 மொழியை தேர்ந்தெடுக்கவும்",
        "tab_predict": "🌱 மகசூல் கணிப்பு",
        "tab_recommend": "💡 பயிர் பரிந்துரை",
        "tab_batch": "📋 தொகுதி கணிப்பு",
        "tab_about": "ℹ️ பற்றி",
        "select_state": "மாநிலம் தேர்ந்தெடுக்கவும்",
        "select_crop": "பயிரை தேர்ந்தெடுக்கவும்",
        "select_season": "பருவத்தை தேர்ந்தெடுக்கவும்",
        "select_year": "ஆண்டை தேர்ந்தெடுக்கவும்",
        "fertilizer": "உரம் (kg/ha)",
        "pesticide": "பூச்சிக்கொல்லி (kg/ha)",
        "soil_header": "🪨 மண் அளவுருக்கள்",
        "nitrogen": "நைட்ரஜன் (N) kg/ha",
        "phosphorus": "பாஸ்பரஸ் (P) kg/ha",
        "potassium": "பொட்டாசியம் (K) kg/ha",
        "soil_ph": "மண் pH",
        "weather_header": "🌤️ வானிலை அளவுருக்கள்",
        "temperature": "சராசரி வெப்பநிலை (°C)",
        "rainfall": "வருடாந்திர மழைவீழ்ச்சி (mm)",
        "humidity": "சராசரி ஈரப்பதம் (%)",
        "lag_header": "📅 வரலாற்று தரவு (விரும்பினால்)",
        "yield_lag1": "கடந்த ஆண்டு மகசூல் (kg/ha)",
        "yield_lag2": "2 ஆண்டுகளுக்கு முன் மகசூல் (kg/ha)",
        "predict_btn": "🔮 மகசூலை கணிக்கவும்",
        "predicted_yield": "கணிக்கப்பட்ட மகசூல்",
        "unit": "kg / ஹெக்டேர்",
        "model_not_found": "⚠️ மாதிரி கோப்பு இல்லை. ML ஸ்கிரிப்டை இயக்கவும்.",
        "result_header": "📊 கணிப்பு முடிவு",
        "good_yield": "✅ சராசரிக்கு மேல் மகசூல்",
        "avg_yield": "⚠️ சராசரி மகசூல்",
        "low_yield": "❌ சராசரிக்கு கீழ் மகசூல்",
        "tips_header": "💡 விவசாய குறிப்புகள்",
        "recommend_header": "🌿 பிராந்திய பயிர் பரிந்துரைகள்",
        "recommend_state": "உங்கள் மாநிலத்தை தேர்ந்தெடுக்கவும்",
        "recommend_season": "பருவம் தேர்ந்தெடுக்கவும்",
        "recommend_btn": "🔍 பரிந்துரைகள் பெறவும்",
        "top_crops": "உங்கள் பகுதிக்கான பரிந்துரைக்கப்பட்ட பயிர்கள்",
        "batch_header": "📋 தொகுதி கணிப்பு",
        "upload_csv": "CSV கோப்பை பதிவேற்றவும்",
        "download_results": "⬇️ முடிவுகளை பதிவிறக்கவும்",
        "about_header": "இந்த செயலியைப் பற்றி",
        "model_stats": "🤖 மாதிரி செயல்திறன்",
        "crop_input_header": "🌾 பயிர் & பருவம்",
        "agri_input_header": "🧪 உரம் & பூச்சிக்கொல்லி",
        "loading": "மகசூல் கணிப்பு கணக்கிடப்படுகிறது...",
        "report_title": "பயிர் மகசூல் கணிப்பு அறிக்கை",
        "report_download": "📥 அறிக்கை பதிவிறக்கவும் (PDF)",
        "chart_soil": "மண் ஊட்டச்சத்து விநியோகம்",
        "chart_trend": "மகசூல் போக்கு பகுப்பாய்வு",
        "chart_weather": "வானிலை மேலோட்டம்",
        "about_dev": "உருவாக்கியவர்",
        "about_model_title": "இயந்திர கற்றல் மாதிரி",
        "about_features_title": "பயன்படுத்தப்பட்ட அம்சங்கள்",
        "about_languages_title": "ஆதரிக்கப்படும் மொழிகள்",
        "about_files_title": "தேவையான கோப்புகள்",
    },
    "తెలుగు (Telugu)": {
        "app_title": "🌾 పంట దిగుబడి అంచనా",
        "app_subtitle": "భారతీయ రైతులకు AI-ఆధారిత వ్యవసాయ మేధస్సు",
        "language_label": "🌐 భాష ఎంచుకోండి",
        "tab_predict": "🌱 దిగుబడి అంచనా",
        "tab_recommend": "💡 పంట సిఫారసు",
        "tab_batch": "📋 బ్యాచ్ అంచనా",
        "tab_about": "ℹ️ గురించి",
        "select_state": "రాష్ట్రం ఎంచుకోండి",
        "select_crop": "పంట ఎంచుకోండి",
        "select_season": "సీజన్ ఎంచుకోండి",
        "select_year": "సంవత్సరం ఎంచుకోండి",
        "fertilizer": "ఎరువు (kg/ha)",
        "pesticide": "పురుగుమందు (kg/ha)",
        "soil_header": "🪨 నేల పారామీటర్లు",
        "nitrogen": "నత్రజని (N) kg/ha",
        "phosphorus": "భాస్వరం (P) kg/ha",
        "potassium": "పొటాషియం (K) kg/ha",
        "soil_ph": "నేల pH",
        "weather_header": "🌤️ వాతావరణ పారామీటర్లు",
        "temperature": "సగటు ఉష్ణోగ్రత (°C)",
        "rainfall": "వార్షిక వర్షపాతం (mm)",
        "humidity": "సగటు తేమ (%)",
        "lag_header": "📅 చారిత్రక డేటా (ఐచ్ఛికం)",
        "yield_lag1": "గత సంవత్సరం దిగుబడి (kg/ha)",
        "yield_lag2": "2 సంవత్సరాల క్రితం దిగుబడి (kg/ha)",
        "predict_btn": "🔮 దిగుబడి అంచనా వేయండి",
        "predicted_yield": "అంచనా దిగుబడి",
        "unit": "kg / హెక్టారు",
        "model_not_found": "⚠️ మోడల్ ఫైల్ కనుగొనబడలేదు. ML స్క్రిప్ట్ రన్ చేయండి.",
        "result_header": "📊 అంచనా ఫలితం",
        "good_yield": "✅ సగటు కంటే ఎక్కువ దిగుబడి",
        "avg_yield": "⚠️ సగటు దిగుబడి",
        "low_yield": "❌ సగటు కంటే తక్కువ దిగుబడి",
        "tips_header": "💡 వ్యవసాయ చిట్కాలు",
        "recommend_header": "🌿 ప్రాంతీయ పంట సిఫారసులు",
        "recommend_state": "మీ రాష్ట్రం ఎంచుకోండి",
        "recommend_season": "సీజన్ ఎంచుకోండి",
        "recommend_btn": "🔍 సిఫారసులు పొందండి",
        "top_crops": "మీ ప్రాంతానికి సిఫారసు చేయబడిన పంటలు",
        "batch_header": "📋 బ్యాచ్ అంచనా",
        "upload_csv": "CSV ఫైల్ అప్‌లోడ్ చేయండి",
        "download_results": "⬇️ ఫలితాలు డౌన్‌లోడ్ చేయండి",
        "about_header": "ఈ యాప్ గురించి",
        "model_stats": "🤖 మోడల్ పనితీరు",
        "crop_input_header": "🌾 పంట & సీజన్",
        "agri_input_header": "🧪 ఎరువు & పురుగుమందు",
        "loading": "దిగుబడి అంచనా లెక్కిస్తున్నాము...",
        "report_title": "పంట దిగుబడి అంచనా నివేదిక",
        "report_download": "📥 నివేదిక డౌన్‌లోడ్ చేయండి (PDF)",
        "chart_soil": "నేల పోషక పంపిణీ",
        "chart_trend": "దిగుబడి ధోరణి విశ్లేషణ",
        "chart_weather": "వాతావరణ అవలోకనం",
        "about_dev": "అభివృద్ధి చేసినవారు",
        "about_model_title": "మెషిన్ లెర్నింగ్ మోడల్",
        "about_features_title": "ఉపయోగించిన లక్షణాలు",
        "about_languages_title": "మద్దతు ఉన్న భాషలు",
        "about_files_title": "అవసరమైన ఫైళ్లు",
    },
    "ಕನ್ನಡ (Kannada)": {
        "app_title": "🌾 ಬೆಳೆ ಇಳುವರಿ ಮುನ್ಸೂಚನೆ",
        "app_subtitle": "ಭಾರತೀಯ ರೈತರಿಗಾಗಿ AI-ಆಧಾರಿತ ಕೃಷಿ ಬುದ್ಧಿ",
        "language_label": "🌐 ಭಾಷೆ ಆಯ್ಕೆ ಮಾಡಿ",
        "tab_predict": "🌱 ಇಳುವರಿ ಮುನ್ಸೂಚನೆ",
        "tab_recommend": "💡 ಬೆಳೆ ಶಿಫಾರಸು",
        "tab_batch": "📋 ಬ್ಯಾಚ್ ಮುನ್ಸೂಚನೆ",
        "tab_about": "ℹ️ ಬಗ್ಗೆ",
        "select_state": "ರಾಜ್ಯ ಆಯ್ಕೆ ಮಾಡಿ",
        "select_crop": "ಬೆಳೆ ಆಯ್ಕೆ ಮಾಡಿ",
        "select_season": "ಋತು ಆಯ್ಕೆ ಮಾಡಿ",
        "select_year": "ವರ್ಷ ಆಯ್ಕೆ ಮಾಡಿ",
        "fertilizer": "ಗೊಬ್ಬರ (kg/ha)",
        "pesticide": "ಕೀಟನಾಶಕ (kg/ha)",
        "soil_header": "🪨 ಮಣ್ಣಿನ ಮಾಪದಂಡಗಳು",
        "nitrogen": "ಸಾರಜನಕ (N) kg/ha",
        "phosphorus": "ರಂಜಕ (P) kg/ha",
        "potassium": "ಪೊಟ್ಯಾಶಿಯಂ (K) kg/ha",
        "soil_ph": "ಮಣ್ಣಿನ pH",
        "weather_header": "🌤️ ಹವಾಮಾನ ಮಾಪದಂಡಗಳು",
        "temperature": "ಸರಾಸರಿ ತಾಪಮಾನ (°C)",
        "rainfall": "ವಾರ್ಷಿಕ ಮಳೆ (mm)",
        "humidity": "ಸರಾಸರಿ ತೇವಾಂಶ (%)",
        "lag_header": "📅 ಐತಿಹಾಸಿಕ ಡೇಟಾ (ಐಚ್ಛಿಕ)",
        "yield_lag1": "ಕಳೆದ ವರ್ಷದ ಇಳುವರಿ (kg/ha)",
        "yield_lag2": "2 ವರ್ಷಗಳ ಹಿಂದಿನ ಇಳುವರಿ (kg/ha)",
        "predict_btn": "🔮 ಇಳುವರಿ ಊಹಿಸಿ",
        "predicted_yield": "ಅಂದಾಜು ಇಳುವರಿ",
        "unit": "kg / ಹೆಕ್ಟೇರ್",
        "model_not_found": "⚠️ ಮಾದರಿ ಫೈಲ್ ಸಿಗಲಿಲ್ಲ. ML ಸ್ಕ್ರಿಪ್ಟ್ ರನ್ ಮಾಡಿ.",
        "result_header": "📊 ಮುನ್ಸೂಚನೆ ಫಲಿತಾಂಶ",
        "good_yield": "✅ ಸರಾಸರಿಗಿಂತ ಹೆಚ್ಚು ಇಳುವರಿ",
        "avg_yield": "⚠️ ಸರಾಸರಿ ಇಳುವರಿ",
        "low_yield": "❌ ಸರಾಸರಿಗಿಂತ ಕಡಿಮೆ ಇಳುವರಿ",
        "tips_header": "💡 ಕೃಷಿ ಸಲಹೆಗಳು",
        "recommend_header": "🌿 ಪ್ರಾದೇಶಿಕ ಬೆಳೆ ಶಿಫಾರಸುಗಳು",
        "recommend_state": "ನಿಮ್ಮ ರಾಜ್ಯ ಆಯ್ಕೆ ಮಾಡಿ",
        "recommend_season": "ಋತು ಆಯ್ಕೆ ಮಾಡಿ",
        "recommend_btn": "🔍 ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಿರಿ",
        "top_crops": "ನಿಮ್ಮ ಪ್ರದೇಶಕ್ಕೆ ಶಿಫಾರಸು ಮಾಡಲಾದ ಬೆಳೆಗಳು",
        "batch_header": "📋 ಬ್ಯಾಚ್ ಮುನ್ಸೂಚನೆ",
        "upload_csv": "CSV ಫೈಲ್ ಅಪ್ಲೋಡ್ ಮಾಡಿ",
        "download_results": "⬇️ ಫಲಿತಾಂಶಗಳನ್ನು ಡೌನ್ಲೋಡ್ ಮಾಡಿ",
        "about_header": "ಈ ಅಪ್ಲಿಕೇಶನ್ ಬಗ್ಗೆ",
        "model_stats": "🤖 ಮಾದರಿ ಕಾರ್ಯಕ್ಷಮತೆ",
        "crop_input_header": "🌾 ಬೆಳೆ & ಋತು",
        "agri_input_header": "🧪 ಗೊಬ್ಬರ & ಕೀಟನಾಶಕ",
        "loading": "ಇಳುವರಿ ಮುನ್ಸೂಚನೆ ಲೆಕ್ಕ ಹಾಕಲಾಗುತ್ತಿದೆ...",
        "report_title": "ಬೆಳೆ ಇಳುವರಿ ಮುನ್ಸೂಚನೆ ವರದಿ",
        "report_download": "📥 ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ (PDF)",
        "chart_soil": "ಮಣ್ಣಿನ ಪೋಷಕ ವಿತರಣೆ",
        "chart_trend": "ಇಳುವರಿ ಟ್ರೆಂಡ್ ವಿಶ್ಲೇಷಣೆ",
        "chart_weather": "ಹವಾಮಾನ ಅವಲೋಕನ",
        "about_dev": "ಅಭಿವೃದ್ಧಿಪಡಿಸಿದವರು",
        "about_model_title": "ಮಶಿನ್ ಲರ್ನಿಂಗ್ ಮಾದರಿ",
        "about_features_title": "ಬಳಸಿದ ವೈಶಿಷ್ಟ್ಯಗಳು",
        "about_languages_title": "ಬೆಂಬಲಿತ ಭಾಷೆಗಳು",
        "about_files_title": "ಅಗತ್ಯ ಫೈಲ್‌ಗಳು",
    },
    "മലയാളം (Malayalam)": {
        "app_title": "🌾 വിള ഉൽപ്പാദന പ്രവചനം",
        "app_subtitle": "ഇന്ത്യൻ കർഷകർക്കായി AI-അധിഷ്ഠിത കാർഷിക ബുദ്ധി",
        "language_label": "🌐 ഭാഷ തിരഞ്ഞെടുക്കുക",
        "tab_predict": "🌱 ഉൽപ്പാദന പ്രവചനം",
        "tab_recommend": "💡 വിള ശുപാർശ",
        "tab_batch": "📋 ബാച്ച് പ്രവചനം",
        "tab_about": "ℹ️ വിവരം",
        "select_state": "സംസ്ഥാനം തിരഞ്ഞെടുക്കുക",
        "select_crop": "വിള തിരഞ്ഞെടുക്കുക",
        "select_season": "സീസൺ തിരഞ്ഞെടുക്കുക",
        "select_year": "വർഷം തിരഞ്ഞെടുക്കുക",
        "fertilizer": "വളം (kg/ha)",
        "pesticide": "കീടനാശിനി (kg/ha)",
        "soil_header": "🪨 മണ്ണ് പാരാമീറ്ററുകൾ",
        "nitrogen": "നൈട്രജൻ (N) kg/ha",
        "phosphorus": "ഫോസ്ഫറസ് (P) kg/ha",
        "potassium": "പൊട്ടാസ്യം (K) kg/ha",
        "soil_ph": "മണ്ണ് pH",
        "weather_header": "🌤️ കാലാവസ്ഥ പാരാമീറ്ററുകൾ",
        "temperature": "ശരാശരി താപനില (°C)",
        "rainfall": "വാർഷിക മഴ (mm)",
        "humidity": "ശരാശരി ആർദ്രത (%)",
        "lag_header": "📅 ചരിത്ര ഡേറ്റ (ഐച്ഛികം)",
        "yield_lag1": "കഴിഞ്ഞ വർഷത്തെ ഉൽപ്പാദനം (kg/ha)",
        "yield_lag2": "2 വർഷം മുൻപ് ഉൽപ്പാദനം (kg/ha)",
        "predict_btn": "🔮 ഉൽപ്പാദനം പ്രവചിക്കുക",
        "predicted_yield": "പ്രവചിക്കപ്പെട്ട ഉൽപ്പാദനം",
        "unit": "kg / ഹെക്ടർ",
        "model_not_found": "⚠️ മോഡൽ ഫയൽ കണ്ടെത്തിയില്ല. ML സ്ക്രിപ്റ്റ് റൺ ചെയ്യുക.",
        "result_header": "📊 പ്രവചന ഫലം",
        "good_yield": "✅ ശരാശരിയേക്കാൾ കൂടുതൽ",
        "avg_yield": "⚠️ ശരാശരി ഉൽപ്പാദനം",
        "low_yield": "❌ ശരാശരിയേക്കാൾ കുറവ്",
        "tips_header": "💡 കൃഷി നുറുങ്ങുകൾ",
        "recommend_header": "🌿 പ്രാദേശിക വിള ശുപാർശകൾ",
        "recommend_state": "നിങ്ങളുടെ സംസ്ഥാനം തിരഞ്ഞെടുക്കുക",
        "recommend_season": "സീസൺ തിരഞ്ഞെടുക്കുക",
        "recommend_btn": "🔍 ശുപാർശകൾ നേടുക",
        "top_crops": "നിങ്ങളുടെ പ്രദേശത്തിന് ശുപാർശ ചെയ്ത വിളകൾ",
        "batch_header": "📋 ബാച്ച് പ്രവചനം",
        "upload_csv": "CSV ഫയൽ അപ്‌ലോഡ് ചെയ്യുക",
        "download_results": "⬇️ ഫലങ്ങൾ ഡൗൺലോഡ് ചെയ്യുക",
        "about_header": "ഈ ആപ്പിനെക്കുറിച്ച്",
        "model_stats": "🤖 മോഡൽ പ്രകടനം",
        "crop_input_header": "🌾 വിള & സീസൺ",
        "agri_input_header": "🧪 വളം & കീടനാശിനി",
        "loading": "ഉൽപ്പാദന പ്രവചനം കണക്കാക്കുന്നു...",
        "report_title": "വിള ഉൽപ്പാദന പ്രവചന റിപ്പോർട്ട്",
        "report_download": "📥 റിപ്പോർട്ട് ഡൗൺലോഡ് ചെയ്യുക (PDF)",
        "chart_soil": "മണ്ണ് പോഷക വിതരണം",
        "chart_trend": "ഉൽപ്പാദന ട്രൻഡ് വിശകലനം",
        "chart_weather": "കാലാവസ്ഥ അവലോകനം",
        "about_dev": "നിർമ്മിച്ചത്",
        "about_model_title": "മഷിൻ ലേണിംഗ് മോഡൽ",
        "about_features_title": "ഉപയോഗിച്ച സവിശേഷതകൾ",
        "about_languages_title": "പിന്തുണയ്ക്കുന്ന ഭാഷകൾ",
        "about_files_title": "ആവശ്യമായ ഫയലുകൾ",
    },
    "ਪੰਜਾਬੀ (Punjabi)": {
        "app_title": "🌾 ਫਸਲ ਉਪਜ ਭਵਿੱਖਬਾਣੀ",
        "app_subtitle": "ਭਾਰਤੀ ਕਿਸਾਨਾਂ ਲਈ AI-ਸੰਚਾਲਿਤ ਖੇਤੀਬਾੜੀ ਬੁੱਧੀ",
        "language_label": "🌐 ਭਾਸ਼ਾ ਚੁਣੋ",
        "tab_predict": "🌱 ਉਪਜ ਭਵਿੱਖਬਾਣੀ",
        "tab_recommend": "💡 ਫਸਲ ਸਿਫਾਰਸ਼",
        "tab_batch": "📋 ਬੈਚ ਭਵਿੱਖਬਾਣੀ",
        "tab_about": "ℹ️ ਬਾਰੇ",
        "select_state": "ਰਾਜ ਚੁਣੋ",
        "select_crop": "ਫਸਲ ਚੁਣੋ",
        "select_season": "ਮੌਸਮ ਚੁਣੋ",
        "select_year": "ਸਾਲ ਚੁਣੋ",
        "fertilizer": "ਖਾਦ (kg/ha)",
        "pesticide": "ਕੀਟਨਾਸ਼ਕ (kg/ha)",
        "soil_header": "🪨 ਮਿੱਟੀ ਦੇ ਪੈਰਾਮੀਟਰ",
        "nitrogen": "ਨਾਈਟ੍ਰੋਜਨ (N) kg/ha",
        "phosphorus": "ਫਾਸਫੋਰਸ (P) kg/ha",
        "potassium": "ਪੋਟਾਸ਼ੀਅਮ (K) kg/ha",
        "soil_ph": "ਮਿੱਟੀ pH",
        "weather_header": "🌤️ ਮੌਸਮ ਪੈਰਾਮੀਟਰ",
        "temperature": "ਔਸਤ ਤਾਪਮਾਨ (°C)",
        "rainfall": "ਸਾਲਾਨਾ ਵਰਖਾ (mm)",
        "humidity": "ਔਸਤ ਨਮੀ (%)",
        "lag_header": "📅 ਇਤਿਹਾਸਕ ਡੇਟਾ (ਵਿਕਲਪਿਕ)",
        "yield_lag1": "ਪਿਛਲੇ ਸਾਲ ਦੀ ਉਪਜ (kg/ha)",
        "yield_lag2": "2 ਸਾਲ ਪਹਿਲਾਂ ਉਪਜ (kg/ha)",
        "predict_btn": "🔮 ਉਪਜ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕਰੋ",
        "predicted_yield": "ਅਨੁਮਾਨਿਤ ਉਪਜ",
        "unit": "kg / ਹੈਕਟੇਅਰ",
        "model_not_found": "⚠️ ਮਾਡਲ ਫਾਈਲ ਨਹੀਂ ਮਿਲੀ। ML ਸਕ੍ਰਿਪਟ ਚਲਾਓ।",
        "result_header": "📊 ਭਵਿੱਖਬਾਣੀ ਨਤੀਜਾ",
        "good_yield": "✅ ਔਸਤ ਤੋਂ ਵੱਧ ਉਪਜ",
        "avg_yield": "⚠️ ਔਸਤ ਉਪਜ",
        "low_yield": "❌ ਔਸਤ ਤੋਂ ਘੱਟ ਉਪਜ",
        "tips_header": "💡 ਖੇਤੀਬਾੜੀ ਸੁਝਾਅ",
        "recommend_header": "🌿 ਖੇਤਰੀ ਫਸਲ ਸਿਫਾਰਸ਼ਾਂ",
        "recommend_state": "ਆਪਣਾ ਰਾਜ ਚੁਣੋ",
        "recommend_season": "ਮੌਸਮ ਚੁਣੋ",
        "recommend_btn": "🔍 ਸਿਫਾਰਸ਼ਾਂ ਪ੍ਰਾਪਤ ਕਰੋ",
        "top_crops": "ਤੁਹਾਡੇ ਖੇਤਰ ਲਈ ਸਿਫਾਰਸ਼ੀ ਫਸਲਾਂ",
        "batch_header": "📋 ਬੈਚ ਭਵਿੱਖਬਾਣੀ",
        "upload_csv": "CSV ਫਾਈਲ ਅਪਲੋਡ ਕਰੋ",
        "download_results": "⬇️ ਨਤੀਜੇ ਡਾਊਨਲੋਡ ਕਰੋ",
        "about_header": "ਇਸ ਐਪ ਬਾਰੇ",
        "model_stats": "🤖 ਮਾਡਲ ਪ੍ਰਦਰਸ਼ਨ",
        "crop_input_header": "🌾 ਫਸਲ & ਮੌਸਮ",
        "agri_input_header": "🧪 ਖਾਦ & ਕੀਟਨਾਸ਼ਕ",
        "loading": "ਉਪਜ ਭਵਿੱਖਬਾਣੀ ਦੀ ਗਣਨਾ ਹੋ ਰਹੀ ਹੈ...",
        "report_title": "ਫਸਲ ਉਪਜ ਭਵਿੱਖਬਾਣੀ ਰਿਪੋਰਟ",
        "report_download": "📥 ਰਿਪੋਰਟ ਡਾਊਨਲੋਡ ਕਰੋ (PDF)",
        "chart_soil": "ਮਿੱਟੀ ਪੋਸ਼ਕ ਵੰਡ",
        "chart_trend": "ਉਪਜ ਰੁਝਾਨ ਵਿਸ਼ਲੇਸ਼ਣ",
        "chart_weather": "ਮੌਸਮ ਝਲਕ",
        "about_dev": "ਦੁਆਰਾ ਵਿਕਸਿਤ",
        "about_model_title": "ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲ",
        "about_features_title": "ਵਰਤੀਆਂ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ",
        "about_languages_title": "ਸਮਰਥਿਤ ਭਾਸ਼ਾਵਾਂ",
        "about_files_title": "ਲੋੜੀਂਦੀਆਂ ਫਾਈਲਾਂ",
    },
    "मराठी (Marathi)": {
        "app_title": "🌾 पीक उत्पादन अंदाज",
        "app_subtitle": "भारतीय शेतकऱ्यांसाठी AI-चालित कृषी बुद्धिमत्ता",
        "language_label": "🌐 भाषा निवडा",
        "tab_predict": "🌱 उत्पादन अंदाज",
        "tab_recommend": "💡 पीक शिफारस",
        "tab_batch": "📋 बॅच अंदाज",
        "tab_about": "ℹ️ माहिती",
        "select_state": "राज्य निवडा",
        "select_crop": "पीक निवडा",
        "select_season": "हंगाम निवडा",
        "select_year": "वर्ष निवडा",
        "fertilizer": "खत (kg/ha)",
        "pesticide": "कीटकनाशक (kg/ha)",
        "soil_header": "🪨 माती पॅरामीटर्स",
        "nitrogen": "नायट्रोजन (N) kg/ha",
        "phosphorus": "फॉस्फरस (P) kg/ha",
        "potassium": "पोटॅशियम (K) kg/ha",
        "soil_ph": "माती pH",
        "weather_header": "🌤️ हवामान पॅरामीटर्स",
        "temperature": "सरासरी तापमान (°C)",
        "rainfall": "वार्षिक पाऊस (mm)",
        "humidity": "सरासरी आर्द्रता (%)",
        "lag_header": "📅 ऐतिहासिक डेटा (पर्यायी)",
        "yield_lag1": "मागील वर्षाचे उत्पादन (kg/ha)",
        "yield_lag2": "२ वर्षांपूर्वी उत्पादन (kg/ha)",
        "predict_btn": "🔮 उत्पादन अंदाज करा",
        "predicted_yield": "अंदाजित उत्पादन",
        "unit": "kg / हेक्टर",
        "model_not_found": "⚠️ मॉडेल फाइल सापडली नाही. ML स्क्रिप्ट चालवा.",
        "result_header": "📊 अंदाज परिणाम",
        "good_yield": "✅ सरासरीपेक्षा जास्त उत्पादन",
        "avg_yield": "⚠️ सरासरी उत्पादन",
        "low_yield": "❌ सरासरीपेक्षा कमी उत्पादन",
        "tips_header": "💡 शेती टिप्स",
        "recommend_header": "🌿 प्रादेशिक पीक शिफारसी",
        "recommend_state": "आपले राज्य निवडा",
        "recommend_season": "हंगाम निवडा",
        "recommend_btn": "🔍 शिफारसी मिळवा",
        "top_crops": "तुमच्या प्रदेशासाठी शिफारस केलेली पिके",
        "batch_header": "📋 बॅच अंदाज",
        "upload_csv": "CSV फाइल अपलोड करा",
        "download_results": "⬇️ परिणाम डाउनलोड करा",
        "about_header": "या अॅपबद्दल",
        "model_stats": "🤖 मॉडेल कामगिरी",
        "crop_input_header": "🌾 पीक & हंगाम",
        "agri_input_header": "🧪 खत & कीटकनाशक",
        "loading": "उत्पादन अंदाजाची गणना होत आहे...",
        "report_title": "पीक उत्पादन अंदाज अहवाल",
        "report_download": "📥 अहवाल डाउनलोड करा (PDF)",
        "chart_soil": "माती पोषक वितरण",
        "chart_trend": "उत्पादन ट्रेंड विश्लेषण",
        "chart_weather": "हवामान आढावा",
        "about_dev": "विकसित केले",
        "about_model_title": "मशीन लर्निंग मॉडेल",
        "about_features_title": "वापरलेली वैशिष्ट्ये",
        "about_languages_title": "समर्थित भाषा",
        "about_files_title": "आवश्यक फाइल्स",
    },
    "ગુજરાતી (Gujarati)": {
        "app_title": "🌾 પાક ઉત્પાદન આગાહી",
        "app_subtitle": "ભારતીય ખેડૂતો માટે AI-સંચાલિત કૃષિ બુદ્ધિ",
        "language_label": "🌐 ભાષા પસંદ કરો",
        "tab_predict": "🌱 ઉત્પાદન આગાહી",
        "tab_recommend": "💡 પાક ભલામણ",
        "tab_batch": "📋 બેચ આગાહી",
        "tab_about": "ℹ️ વિષે",
        "select_state": "રાજ્ય પસંદ કરો",
        "select_crop": "પાક પસંદ કરો",
        "select_season": "સીઝન પસંદ કરો",
        "select_year": "વર્ષ પસંદ કરો",
        "fertilizer": "ખાતર (kg/ha)",
        "pesticide": "જંતુનાશક (kg/ha)",
        "soil_header": "🪨 માટી પ્રમાણ",
        "nitrogen": "નાઇટ્રોજન (N) kg/ha",
        "phosphorus": "ફોસ્ફરસ (P) kg/ha",
        "potassium": "પોટેશિયમ (K) kg/ha",
        "soil_ph": "માટી pH",
        "weather_header": "🌤️ હવામાન પ્રમાણ",
        "temperature": "સરેરાશ તાપમાન (°C)",
        "rainfall": "વાર્ષિક વરસાદ (mm)",
        "humidity": "સરેરાશ ભેજ (%)",
        "lag_header": "📅 ઐતિહાસિક ડેટા (વૈકલ્પિક)",
        "yield_lag1": "ગત વર્ષ ઉત્પાદન (kg/ha)",
        "yield_lag2": "2 વર્ષ પહેલાં ઉત્પાદન (kg/ha)",
        "predict_btn": "🔮 ઉત્પાદન આગાહી કરો",
        "predicted_yield": "અંદાજિત ઉત્પાદન",
        "unit": "kg / હેક્ટર",
        "model_not_found": "⚠️ મોડેલ ફાઇલ મળી નથી. ML સ્ક્રિપ્ટ ચલાવો.",
        "result_header": "📊 આગાહી પરિણામ",
        "good_yield": "✅ સરેરાશ કરતાં વધુ ઉત્પાદન",
        "avg_yield": "⚠️ સરેરાશ ઉત્પાદન",
        "low_yield": "❌ સરેરાશ કરતાં ઓછું ઉત્પાદન",
        "tips_header": "💡 ખેતી ટિપ્સ",
        "recommend_header": "🌿 પ્રાદેશિક પાક ભલામણો",
        "recommend_state": "તમારું રાજ્ય પસંદ કરો",
        "recommend_season": "સીઝન પસંદ કરો",
        "recommend_btn": "🔍 ભલામણો મેળવો",
        "top_crops": "તમારા ક્ષેત્ર માટે ભલામણ કરેલ પાક",
        "batch_header": "📋 બેચ આગાહી",
        "upload_csv": "CSV ફાઇલ અપલોડ કરો",
        "download_results": "⬇️ પરિણામો ડાઉનલોડ કરો",
        "about_header": "આ એપ વિષે",
        "model_stats": "🤖 મોડેલ પ્રદર્શન",
        "crop_input_header": "🌾 પાક & સીઝન",
        "agri_input_header": "🧪 ખાતર & જંતુનાશક",
        "loading": "ઉત્પાદન આગાહી ગણવામાં આવી રહી છે...",
        "report_title": "પાક ઉત્પાદન આગાહી અહેવાલ",
        "report_download": "📥 અહેવાલ ડાઉનલોડ કરો (PDF)",
        "chart_soil": "માટી પોષક વિતરણ",
        "chart_trend": "ઉત્પાદન વલણ વિશ્લેષણ",
        "chart_weather": "હવામાન સારાંશ",
        "about_dev": "દ્વારા વિકસિત",
        "about_model_title": "મશીન લર્નિંગ મોડેલ",
        "about_features_title": "ઉપયોગ કરેલ સુવિધાઓ",
        "about_languages_title": "સમર્થિત ભાષાઓ",
        "about_files_title": "જરૂરી ફાઈલો",
    },
    "ଓଡ଼ିଆ (Odia)": {
        "app_title": "🌾 ଫସଲ ଅମଳ ପୂର୍ବାନୁମାନ",
        "app_subtitle": "ଭାରତୀୟ କୃଷକମାନଙ୍କ ପାଇଁ AI-ଚାଳିତ କୃଷି ବୁଦ୍ଧି",
        "language_label": "🌐 ଭାଷା ବାଛନ୍ତୁ",
        "tab_predict": "🌱 ଅମଳ ପୂର୍ବାନୁମାନ",
        "tab_recommend": "💡 ଫସଲ ସୁପାରିଶ",
        "tab_batch": "📋 ବ୍ୟାଚ ପୂର୍ବାନୁମାନ",
        "tab_about": "ℹ️ ବିଷୟରେ",
        "select_state": "ରାଜ୍ୟ ବାଛନ୍ତୁ",
        "select_crop": "ଫସଲ ବାଛନ୍ତୁ",
        "select_season": "ଋତୁ ବାଛନ୍ତୁ",
        "select_year": "ବର୍ଷ ବାଛନ୍ତୁ",
        "fertilizer": "ସାର (kg/ha)",
        "pesticide": "କୀଟନାଶକ (kg/ha)",
        "soil_header": "🪨 ମୃତ୍ତିକା ପ୍ରଚଳ",
        "nitrogen": "ନାଇଟ୍ରୋଜେନ (N) kg/ha",
        "phosphorus": "ଫସଫରସ (P) kg/ha",
        "potassium": "ପୋଟାସିୟମ (K) kg/ha",
        "soil_ph": "ମୃତ୍ତିକା pH",
        "weather_header": "🌤️ ପାଣିପାଗ ପ୍ରଚଳ",
        "temperature": "ହାରାହାରି ତାପମାତ୍ରା (°C)",
        "rainfall": "ବାର୍ଷିକ ବର୍ଷା (mm)",
        "humidity": "ହାରାହାରି ଆର୍ଦ୍ରତା (%)",
        "lag_header": "📅 ଐତିହାସିକ ତଥ୍ୟ (ଐଚ୍ଛିକ)",
        "yield_lag1": "ଗତ ବର୍ଷ ଅମଳ (kg/ha)",
        "yield_lag2": "2 ବର୍ଷ ପୂର୍ବ ଅମଳ (kg/ha)",
        "predict_btn": "🔮 ଅମଳ ପୂର୍ବାନୁମାନ କରନ୍ତୁ",
        "predicted_yield": "ଆନୁମାନିକ ଅମଳ",
        "unit": "kg / ହେକ୍ଟର",
        "model_not_found": "⚠️ ମଡେଲ ଫାଇଲ ମିଳିଲା ନାହିଁ। ML ସ୍କ୍ରିପ୍ଟ ଚଲାନ୍ତୁ।",
        "result_header": "📊 ପୂର୍ବାନୁମାନ ଫଳ",
        "good_yield": "✅ ହାରାହାରିଠାରୁ ଅଧିକ ଅମଳ",
        "avg_yield": "⚠️ ହାରାହାରି ଅମଳ",
        "low_yield": "❌ ହାରାହାରିଠାରୁ କମ ଅମଳ",
        "tips_header": "💡 କୃଷି ଟିପ୍ସ",
        "recommend_header": "🌿 ଆଞ୍ଚଳିକ ଫସଲ ସୁପାରିଶ",
        "recommend_state": "ଆପଣଙ୍କ ରାଜ୍ୟ ବାଛନ୍ତୁ",
        "recommend_season": "ଋତୁ ବାଛନ୍ତୁ",
        "recommend_btn": "🔍 ସୁପାରିଶ ପ୍ରାପ୍ତ କରନ୍ତୁ",
        "top_crops": "ଆପଣଙ୍କ ଅଞ୍ଚଳ ପାଇଁ ସୁପାରିଶ ଫସଲ",
        "batch_header": "📋 ବ୍ୟାଚ ପୂର୍ବାନୁମାନ",
        "upload_csv": "CSV ଫାଇଲ ଅପଲୋଡ କରନ୍ତୁ",
        "download_results": "⬇️ ଫଳ ଡାଉନଲୋଡ କରନ୍ତୁ",
        "about_header": "ଏହି ଆପ ବିଷୟରେ",
        "model_stats": "🤖 ମଡେଲ କାର୍ଯ୍ୟକ୍ଷମତା",
        "crop_input_header": "🌾 ଫସଲ & ଋତୁ",
        "agri_input_header": "🧪 ସାର & କୀଟନାଶକ",
        "loading": "ଅମଳ ପୂର୍ବାନୁମାନ ଗଣନା ହେଉଛି...",
        "report_title": "ଫସଲ ଅମଳ ପୂର୍ବାନୁମାନ ରିପୋର୍ଟ",
        "report_download": "📥 ରିପୋର୍ଟ ଡାଉନଲୋଡ କରନ୍ତୁ (PDF)",
        "chart_soil": "ମୃତ୍ତିକା ପୋଷଣ ବଣ୍ଟନ",
        "chart_trend": "ଅମଳ ଧାରା ବିଶ୍ଳେଷଣ",
        "chart_weather": "ପାଣିପାଗ ସାର",
        "about_dev": "ଦ୍ୱାରା ବିକଶିତ",
        "about_model_title": "ମଶିନ ଲର୍ନିଂ ମଡେଲ",
        "about_features_title": "ବ୍ୟବହୃତ ବୈଶିଷ୍ଟ୍ୟ",
        "about_languages_title": "ସମର୍ଥିତ ଭାଷା",
        "about_files_title": "ଆବଶ୍ୟକ ଫାଇଲ",
    },
}

# ──── URDU ────
TRANSLATIONS["اردو (Urdu)"] = {
    "app_title": "🌾 فصل پیداوار پیش گوئی",
    "app_subtitle": "ہندوستانی کسانوں کے لیے AI زراعت",
    "language_label": "🌐 زبان منتخب کریں",
    "tab_predict": "🌱 پیداوار پیش گوئی",
    "tab_recommend": "💡 فصل کی سفارش",
    "tab_batch": "📋 بیچ پیش گوئی",
    "tab_about": "ℹ️ کے بارے میں",
    "select_state": "ریاست منتخب کریں",
    "select_crop": "فصل منتخب کریں",
    "select_season": "موسم منتخب کریں",
    "select_year": "سال منتخب کریں",
    "fertilizer": "کھاد (kg/ha)",
    "pesticide": "کیڑے مار دوا (kg/ha)",
    "soil_header": "🪨 مٹی کے پیرامیٹر",
    "nitrogen": "نائٹروجن (N) kg/ha",
    "phosphorus": "فاسفورس (P) kg/ha",
    "potassium": "پوٹاشیم (K) kg/ha",
    "soil_ph": "مٹی pH",
    "weather_header": "🌤️ موسمی پیرامیٹر",
    "temperature": "اوسط درجہ حرارت (°C)",
    "rainfall": "سالانہ بارش (mm)",
    "humidity": "اوسط نمی (%)",
    "lag_header": "📅 تاریخی ڈیٹا (اختیاری)",
    "yield_lag1": "گزشتہ سال کی پیداوار (kg/ha)",
    "yield_lag2": "2 سال پہلے کی پیداوار (kg/ha)",
    "predict_btn": "🔮 پیداوار کی پیش گوئی کریں",
    "predicted_yield": "متوقع پیداوار",
    "unit": "kg / ہیکٹیئر",
    "model_not_found": "⚠️ ماڈل فائل نہیں ملی۔ ML اسکرپٹ چلائیں۔",
    "result_header": "📊 پیش گوئی کا نتیجہ",
    "good_yield": "✅ اوسط سے زیادہ پیداوار",
    "avg_yield": "⚠️ اوسط پیداوار",
    "low_yield": "❌ اوسط سے کم پیداوار",
    "tips_header": "💡 زراعتی تجاویز",
    "recommend_header": "🌿 علاقائی فصل کی سفارشات",
    "recommend_state": "اپنی ریاست منتخب کریں",
    "recommend_season": "موسم منتخب کریں",
    "recommend_btn": "🔍 سفارشات حاصل کریں",
    "top_crops": "آپ کے علاقے کے لیے تجویز کردہ فصلیں",
    "batch_header": "📋 بیچ پیش گوئی",
    "upload_csv": "CSV فائل اپ لوڈ کریں",
    "download_results": "⬇️ نتائج ڈاؤن لوڈ کریں",
    "about_header": "اس ایپ کے بارے میں",
    "model_stats": "🤖 ماڈل کارکردگی",
    "crop_input_header": "🌾 فصل اور موسم",
    "agri_input_header": "🧪 کھاد اور کیڑے مار دوا",
    "loading": "پیداوار پیش گوئی کا حساب لگایا جا رہا ہے...",
    "report_title": "فصل پیداوار رپورٹ",
    "report_download": "📥 رپورٹ ڈاؤن لوڈ کریں (PDF)",
    "chart_soil": "مٹی غذائیت تقسیم",
    "chart_trend": "پیداوار رجحان",
    "chart_weather": "موسمی جائزہ",
    "about_dev": "بنایا از",
    "about_model_title": "مشین لرننگ ماڈل",
    "about_features_title": "استعمال شدہ خصوصیات",
    "about_languages_title": "معاون زبانیں",
    "about_files_title": "ضروری فائلیں",
}
# All remaining 8th Schedule Indian languages

TRANSLATIONS["অসমীয়া (Assamese)"] = {
    "app_title": "🌾 শস্য উৎপাদন পূৰ্বানুমান",
    "app_subtitle": "ভাৰতীয় কৃষকসকলৰ বাবে AI-চালিত কৃষি বুদ্ধি",
    "language_label": "🌐 ভাষা বাছনি কৰক",
    "tab_predict": "🌱 উৎপাদন পূৰ্বানুমান",
    "tab_recommend": "💡 শস্য পৰামৰ্শ",
    "tab_batch": "📋 বেচ পূৰ্বানুমান",
    "tab_about": "ℹ️ বিষয়ে",
    "select_state": "ৰাজ্য বাছনি কৰক",
    "select_crop": "শস্য বাছনি কৰক",
    "select_season": "বতৰ বাছনি কৰক",
    "select_year": "বছৰ বাছনি কৰক",
    "fertilizer": "সাৰ (kg/ha)",
    "pesticide": "কীটনাশক (kg/ha)",
    "soil_header": "🪨 মাটিৰ প্ৰাচল",
    "nitrogen": "নাইট্ৰ'জেন (N) kg/ha",
    "phosphorus": "ফচফৰাছ (P) kg/ha",
    "potassium": "পটাছিয়াম (K) kg/ha",
    "soil_ph": "মাটিৰ pH",
    "weather_header": "🌤️ বতৰৰ প্ৰাচল",
    "temperature": "গড় উষ্ণতা (°C)",
    "rainfall": "বাৰ্ষিক বৰষুণ (mm)",
    "humidity": "গড় আৰ্দ্ৰতা (%)",
    "lag_header": "📅 ঐতিহাসিক তথ্য (ঐচ্ছিক)",
    "yield_lag1": "যোৱা বছৰৰ উৎপাদন (kg/ha)",
    "yield_lag2": "২ বছৰ আগৰ উৎপাদন (kg/ha)",
    "predict_btn": "🔮 উৎপাদন পূৰ্বানুমান কৰক",
    "predicted_yield": "পূৰ্বানুমানিত উৎপাদন",
    "unit": "kg / হেক্টৰ",
    "model_not_found": "⚠️ মডেল ফাইল পোৱা নগ'ল। ML স্ক্ৰিপ্ট চলাওক।",
    "result_header": "📊 পূৰ্বানুমানৰ ফলাফল",
    "good_yield": "✅ গড়তকৈ বেছি উৎপাদন",
    "avg_yield": "⚠️ গড় উৎপাদন",
    "low_yield": "❌ গড়তকৈ কম উৎপাদন",
    "tips_header": "💡 কৃষি পৰামৰ্শ",
    "recommend_header": "🌿 আঞ্চলিক শস্য পৰামৰ্শ",
    "recommend_state": "আপোনাৰ ৰাজ্য বাছনি কৰক",
    "recommend_season": "বতৰ বাছনি কৰক",
    "recommend_btn": "🔍 পৰামৰ্শ লাভ কৰক",
    "top_crops": "আপোনাৰ অঞ্চলৰ বাবে পৰামৰ্শিত শস্য",
    "batch_header": "📋 বেচ পূৰ্বানুমান",
    "upload_csv": "CSV ফাইল আপলোড কৰক",
    "download_results": "⬇️ ফলাফল ডাউনলোড কৰক",
    "about_header": "এই এপৰ বিষয়ে",
    "model_stats": "🤖 মডেল কাৰ্যক্ষমতা",
    "crop_input_header": "🌾 শস্য আৰু বতৰ",
    "agri_input_header": "🧪 সাৰ আৰু কীটনাশক",
    "loading": "উৎপাদন পূৰ্বানুমান গণনা হৈছে...",
    "report_title": "শস্য উৎপাদন পূৰ্বানুমান প্ৰতিবেদন",
    "report_download": "📥 প্ৰতিবেদন ডাউনলোড কৰক",
    "chart_soil": "মাটিৰ পুষ্টি বিতৰণ",
    "chart_trend": "উৎপাদন প্ৰৱণতা",
    "chart_weather": "বতৰৰ সাৰাংশ",
    "about_dev": "দ্বাৰা বিকশিত",
    "about_model_title": "মেচিন লাৰ্নিং মডেল",
    "about_features_title": "ব্যৱহৃত বৈশিষ্ট্য",
    "about_languages_title": "সমৰ্থিত ভাষা",
    "about_files_title": "প্ৰয়োজনীয় ফাইল",
}

TRANSLATIONS["मैथिली (Maithili)"] = {
    "app_title": "🌾 फसल उत्पादन अनुमान",
    "app_subtitle": "भारतीय किसानक लेल AI-संचालित कृषि बुद्धि",
    "language_label": "🌐 भाषा चुनू",
    "tab_predict": "🌱 उत्पादन अनुमान",
    "tab_recommend": "💡 फसल सिफारिश",
    "tab_batch": "📋 बैच अनुमान",
    "tab_about": "ℹ️ जानकारी",
    "select_state": "राज्य चुनू",
    "select_crop": "फसल चुनू",
    "select_season": "मौसम चुनू",
    "select_year": "साल चुनू",
    "fertilizer": "खाद (kg/ha)",
    "pesticide": "कीटनाशक (kg/ha)",
    "soil_header": "🪨 माटिक प्राचल",
    "nitrogen": "नाइट्रोजन (N) kg/ha",
    "phosphorus": "फॉस्फोरस (P) kg/ha",
    "potassium": "पोटेशियम (K) kg/ha",
    "soil_ph": "माटिक pH",
    "weather_header": "🌤️ मौसमक प्राचल",
    "temperature": "औसत तापमान (°C)",
    "rainfall": "वार्षिक वर्षा (mm)",
    "humidity": "औसत आर्द्रता (%)",
    "lag_header": "📅 ऐतिहासिक डेटा (वैकल्पिक)",
    "yield_lag1": "पिछला साल उत्पादन (kg/ha)",
    "yield_lag2": "2 साल पहिने उत्पादन (kg/ha)",
    "predict_btn": "🔮 उत्पादन अनुमान करू",
    "predicted_yield": "अनुमानित उत्पादन",
    "unit": "kg / हेक्टेयर",
    "model_not_found": "⚠️ मॉडल फाइल नहि भेटल। ML स्क्रिप्ट चलाउ।",
    "result_header": "📊 अनुमानक परिणाम",
    "good_yield": "✅ औसतसँ बेशी उत्पादन",
    "avg_yield": "⚠️ औसत उत्पादन",
    "low_yield": "❌ औसतसँ कम उत्पादन",
    "tips_header": "💡 खेती सुझाव",
    "recommend_header": "🌿 क्षेत्रीय फसल सिफारिश",
    "recommend_state": "अपन राज्य चुनू",
    "recommend_season": "मौसम चुनू",
    "recommend_btn": "🔍 सिफारिश पाउ",
    "top_crops": "अहाँक क्षेत्रक लेल अनुशंसित फसल",
    "batch_header": "📋 बैच अनुमान",
    "upload_csv": "CSV फाइल अपलोड करू",
    "download_results": "⬇️ परिणाम डाउनलोड करू",
    "about_header": "एहि एप्पक बारेमे",
    "model_stats": "🤖 मॉडल प्रदर्शन",
    "crop_input_header": "🌾 फसल आ मौसम",
    "agri_input_header": "🧪 खाद आ कीटनाशक",
    "loading": "उत्पादन अनुमानक गणना भ रहल अछि...",
    "report_title": "फसल उत्पादन अनुमान रिपोर्ट",
    "report_download": "📥 रिपोर्ट डाउनलोड करू",
    "chart_soil": "माटिक पोषण वितरण",
    "chart_trend": "उत्पादन प्रवृत्ति",
    "chart_weather": "मौसम सारांश",
    "about_dev": "द्वारा विकसित",
    "about_model_title": "मशीन लर्निंग मॉडल",
    "about_features_title": "प्रयुक्त विशेषता",
    "about_languages_title": "समर्थित भाषा",
    "about_files_title": "आवश्यक फाइल",
}

TRANSLATIONS["کٲشُر (Kashmiri)"] = {
    "app_title": "🌾 فصل پیداوار اندازہ",
    "app_subtitle": "ہندوستانی کسانہِ ہوِتہٕ AI زراعت",
    "language_label": "🌐 زبان ہیٹھ کٔرِو",
    "tab_predict": "🌱 پیداوار اندازہ",
    "tab_recommend": "💡 فصل صلاح",
    "tab_batch": "📋 بیچ اندازہ",
    "tab_about": "ℹ️ بابتہٕ",
    "select_state": "صوبہ ہیٹھ کٔرِو",
    "select_crop": "فصل ہیٹھ کٔرِو",
    "select_season": "موسم ہیٹھ کٔرِو",
    "select_year": "ورہہ ہیٹھ کٔرِو",
    "fertilizer": "کھاد (kg/ha)",
    "pesticide": "کیڑے مار (kg/ha)",
    "soil_header": "🪨 مٲٹی معلومات",
    "nitrogen": "نائٹروجن (N) kg/ha",
    "phosphorus": "فاسفورس (P) kg/ha",
    "potassium": "پوٹاشیم (K) kg/ha",
    "soil_ph": "مٲٹی pH",
    "weather_header": "🌤️ موسمی معلومات",
    "temperature": "اوسط گرمی (°C)",
    "rainfall": "سالانہ بارش (mm)",
    "humidity": "اوسط نمی (%)",
    "lag_header": "📅 پرانہٕ ڈیٹا (اختیاری)",
    "yield_lag1": "گژھٕ ورہہ پیداوار (kg/ha)",
    "yield_lag2": "2 ورہہ پیٹھ پیداوار (kg/ha)",
    "predict_btn": "🔮 پیداوار اندازہ کٔرِو",
    "predicted_yield": "متوقع پیداوار",
    "unit": "kg / ہیکٹیئر",
    "model_not_found": "⚠️ ماڈل فائل نہٕ ملیہٕ۔",
    "result_header": "📊 اندازہٕ نتیجہ",
    "good_yield": "✅ اوسطہٕ زیادہ پیداوار",
    "avg_yield": "⚠️ اوسط پیداوار",
    "low_yield": "❌ اوسطہٕ کم پیداوار",
    "tips_header": "💡 زراعتی مشورہ",
    "recommend_header": "🌿 علاقائی فصل صلاح",
    "recommend_state": "اپنہٕ صوبہ ہیٹھ کٔرِو",
    "recommend_season": "موسم ہیٹھ کٔرِو",
    "recommend_btn": "🔍 صلاح حاصل کٔرِو",
    "top_crops": "تہٕندِ علاقہٕ ہوِتہٕ فصل",
    "batch_header": "📋 بیچ اندازہ",
    "upload_csv": "CSV فائل اپلوڈ کٔرِو",
    "download_results": "⬇️ نتیجہ ڈاؤنلوڈ کٔرِو",
    "about_header": "ایپ بابتہٕ",
    "model_stats": "🤖 ماڈل کارکردگی",
    "crop_input_header": "🌾 فصل آ موسم",
    "agri_input_header": "🧪 کھاد آ کیڑے مار",
    "loading": "پیداوار اندازہ حساب...",
    "report_title": "فصل پیداوار اندازہ رپورٹ",
    "report_download": "📥 رپورٹ ڈاؤنلوڈ کٔرِو",
    "chart_soil": "مٲٹی غذائیت",
    "chart_trend": "پیداوار رجحان",
    "chart_weather": "موسم جائزہ",
    "about_dev": "بنایہٕ",
    "about_model_title": "مشین لرننگ ماڈل",
    "about_features_title": "استعمال خصوصیات",
    "about_languages_title": "زبانہٕ",
    "about_files_title": "فائلہٕ",
}

TRANSLATIONS["मणिपुरी (Meitei)"] = {
    "app_title": "🌾 হাও থাজবা ওইবা চাউখৎপা",
    "app_subtitle": "ভারতকী লৈঙাক্লোনগী থৌদাং পুম্নমক AI পাওখুম",
    "language_label": "🌐 লোন থুং্নবা",
    "tab_predict": "🌱 চাউখৎপা অনুমান",
    "tab_recommend": "💡 হাও কনখৎপা",
    "tab_batch": "📋 বেচ অনুমান",
    "tab_about": "ℹ️ খন্নবা",
    "select_state": "স্তেট থুং্নবা",
    "select_crop": "হাও থুং্নবা",
    "select_season": "ইরাং থুং্নবা",
    "select_year": "চহী থুং্নবা",
    "fertilizer": "ফর্তিলাইজর (kg/ha)",
    "pesticide": "পেস্তিসাইড (kg/ha)",
    "soil_header": "🪨 লৈবাক পারামিতর",
    "nitrogen": "নাইত্রোজেন (N) kg/ha",
    "phosphorus": "ফস্ফোরস (P) kg/ha",
    "potassium": "পোতাসিয়ম (K) kg/ha",
    "soil_ph": "লৈবাক pH",
    "weather_header": "🌤️ তাইবং পারামিতর",
    "temperature": "অভরেজ তেম্পরেচর (°C)",
    "rainfall": "চহী মরম (mm)",
    "humidity": "অভরেজ হুমিদিতি (%)",
    "lag_header": "📅 মখোয় ডেতা (ঐচ্ছিক)",
    "yield_lag1": "মখোয় চহীগী চাউখৎপা (kg/ha)",
    "yield_lag2": "চহী ২গী মখোয় চাউখৎপা (kg/ha)",
    "predict_btn": "🔮 চাউখৎপা অনুমান তৌবা",
    "predicted_yield": "অনুমান চাউখৎপা",
    "unit": "kg / হেক্তর",
    "model_not_found": "⚠️ মোদেল ফাইল ফংলমে। ML স্ক্রিপ্ত তান্নবা।",
    "result_header": "📊 অনুমানগী ফলাফল",
    "good_yield": "✅ অভরেজকদা মথৌ চাউখৎপা",
    "avg_yield": "⚠️ অভরেজ চাউখৎপা",
    "low_yield": "❌ অভরেজকদা কম চাউখৎপা",
    "tips_header": "💡 লৈবাক তিপস",
    "recommend_header": "🌿 রিজিওনেল হাও কনখৎপা",
    "recommend_state": "নংগী স্তেট থুং্নবা",
    "recommend_season": "ইরাং থুং্নবা",
    "recommend_btn": "🔍 কনখৎপা পাওবা",
    "top_crops": "নংগী এরিয়াগী হাও",
    "batch_header": "📋 বেচ অনুমান",
    "upload_csv": "CSV ফাইল আপলোড তৌবা",
    "download_results": "⬇️ ফলাফল ডাউনলোড তৌবা",
    "about_header": "এপ খন্নবা",
    "model_stats": "🤖 মোদেল পর্ফোর্মেন্স",
    "crop_input_header": "🌾 হাও আমসুং ইরাং",
    "agri_input_header": "🧪 ফর্তিলাইজর আমসুং পেস্তিসাইড",
    "loading": "চাউখৎপা অনুমান হন্থহনবা...",
    "report_title": "হাও চাউখৎপা অনুমান রিপোর্ট",
    "report_download": "📥 রিপোর্ট ডাউনলোড তৌবা",
    "chart_soil": "লৈবাক নুত্রিএন্ত",
    "chart_trend": "চাউখৎপা ত্রেন্ড",
    "chart_weather": "তাইবং সারাংস",
    "about_dev": "ৱারেপ তৌখিবা",
    "about_model_title": "মেশিন লর্নিং মোদেল",
    "about_features_title": "ফিচর",
    "about_languages_title": "লোন",
    "about_files_title": "ফাইল",
}

TRANSLATIONS["नेपाली (Nepali)"] = {
    "app_title": "🌾 बाली उत्पादन भविष्यवाणी",
    "app_subtitle": "भारतीय किसानहरूका लागि AI-संचालित कृषि बुद्धि",
    "language_label": "🌐 भाषा छान्नुहोस्",
    "tab_predict": "🌱 उत्पादन भविष्यवाणी",
    "tab_recommend": "💡 बाली सिफारिस",
    "tab_batch": "📋 ब्याच भविष्यवाणी",
    "tab_about": "ℹ️ बारेमा",
    "select_state": "राज्य छान्नुहोस्",
    "select_crop": "बाली छान्नुहोस्",
    "select_season": "मौसम छान्नुहोस्",
    "select_year": "वर्ष छान्नुहोस्",
    "fertilizer": "मल (kg/ha)",
    "pesticide": "कीटनाशक (kg/ha)",
    "soil_header": "🪨 माटो प्यारामिटर",
    "nitrogen": "नाइट्रोजन (N) kg/ha",
    "phosphorus": "फस्फोरस (P) kg/ha",
    "potassium": "पोटासियम (K) kg/ha",
    "soil_ph": "माटो pH",
    "weather_header": "🌤️ मौसम प्यारामिटर",
    "temperature": "औसत तापक्रम (°C)",
    "rainfall": "वार्षिक वर्षा (mm)",
    "humidity": "औसत आर्द्रता (%)",
    "lag_header": "📅 ऐतिहासिक डेटा (ऐच्छिक)",
    "yield_lag1": "गत वर्षको उत्पादन (kg/ha)",
    "yield_lag2": "२ वर्ष अघिको उत्पादन (kg/ha)",
    "predict_btn": "🔮 उत्पादन भविष्यवाणी गर्नुहोस्",
    "predicted_yield": "अनुमानित उत्पादन",
    "unit": "kg / हेक्टेयर",
    "model_not_found": "⚠️ मोडेल फाइल फेला परेन। ML स्क्रिप्ट चलाउनुहोस्।",
    "result_header": "📊 भविष्यवाणी परिणाम",
    "good_yield": "✅ औसतभन्दा बढी उत्पादन",
    "avg_yield": "⚠️ औसत उत्पादन",
    "low_yield": "❌ औसतभन्दा कम उत्पादन",
    "tips_header": "💡 कृषि सुझाव",
    "recommend_header": "🌿 क्षेत्रीय बाली सिफारिस",
    "recommend_state": "आफ्नो राज्य छान्नुहोस्",
    "recommend_season": "मौसम छान्नुहोस्",
    "recommend_btn": "🔍 सिफारिस पाउनुहोस्",
    "top_crops": "तपाईंको क्षेत्रका लागि सिफारिस गरिएका बाली",
    "batch_header": "📋 ब्याच भविष्यवाणी",
    "upload_csv": "CSV फाइल अपलोड गर्नुहोस्",
    "download_results": "⬇️ परिणाम डाउनलोड गर्नुहोस्",
    "about_header": "यो एपबारे",
    "model_stats": "🤖 मोडेल प्रदर्शन",
    "crop_input_header": "🌾 बाली र मौसम",
    "agri_input_header": "🧪 मल र कीटनाशक",
    "loading": "उत्पादन भविष्यवाणी गणना हुँदैछ...",
    "report_title": "बाली उत्पादन भविष्यवाणी प्रतिवेदन",
    "report_download": "📥 प्रतिवेदन डाउनलोड गर्नुहोस्",
    "chart_soil": "माटो पोषण वितरण",
    "chart_trend": "उत्पादन प्रवृत्ति",
    "chart_weather": "मौसम सारांश",
    "about_dev": "द्वारा विकसित",
    "about_model_title": "मेशिन लर्निंग मोडेल",
    "about_features_title": "प्रयोग गरिएका विशेषताहरू",
    "about_languages_title": "समर्थित भाषाहरू",
    "about_files_title": "आवश्यक फाइलहरू",
}

TRANSLATIONS["सिन्धी (Sindhi)"] = {
    "app_title": "🌾 فصل پيداوار اندازو",
    "app_subtitle": "هندستاني ڪسانن لاءِ AI زراعت",
    "language_label": "🌐 ٻولي چونڊيو",
    "tab_predict": "🌱 پيداوار اندازو",
    "tab_recommend": "💡 فصل صلاح",
    "tab_batch": "📋 بيچ اندازو",
    "tab_about": "ℹ️ باري",
    "select_state": "صوبو چونڊيو",
    "select_crop": "فصل چونڊيو",
    "select_season": "موسم چونڊيو",
    "select_year": "سال چونڊيو",
    "fertilizer": "کاد (kg/ha)",
    "pesticide": "ڪيڙي مار (kg/ha)",
    "soil_header": "🪨 مٽي جا پيٽاڻ",
    "nitrogen": "نائيٽروجن (N) kg/ha",
    "phosphorus": "فاسفورس (P) kg/ha",
    "potassium": "پوٽاشيم (K) kg/ha",
    "soil_ph": "مٽي pH",
    "weather_header": "🌤️ موسمي پيٽاڻ",
    "temperature": "اوسط درجه حرارت (°C)",
    "rainfall": "سالياني بارش (mm)",
    "humidity": "اوسط نمي (%)",
    "lag_header": "📅 پراڻو ڊيٽا (اختياري)",
    "yield_lag1": "پوئين سال پيداوار (kg/ha)",
    "yield_lag2": "2 سال اڳ پيداوار (kg/ha)",
    "predict_btn": "🔮 پيداوار اندازو ڪريو",
    "predicted_yield": "متوقع پيداوار",
    "unit": "kg / هيڪٽيئر",
    "model_not_found": "⚠️ ماڊل فائل نه مليو۔ ML اسڪرپٽ هلايو۔",
    "result_header": "📊 اندازي جو نتيجو",
    "good_yield": "✅ اوسط کان وڌيڪ پيداوار",
    "avg_yield": "⚠️ اوسط پيداوار",
    "low_yield": "❌ اوسط کان گهٽ پيداوار",
    "tips_header": "💡 زراعتي مشورو",
    "recommend_header": "🌿 علائقائي فصل صلاح",
    "recommend_state": "پنهنجو صوبو چونڊيو",
    "recommend_season": "موسم چونڊيو",
    "recommend_btn": "🔍 صلاح حاصل ڪريو",
    "top_crops": "توهان جي علائقي لاءِ فصل",
    "batch_header": "📋 بيچ اندازو",
    "upload_csv": "CSV فائل اپلوڊ ڪريو",
    "download_results": "⬇️ نتيجا ڊائونلوڊ ڪريو",
    "about_header": "هن ايپ باري",
    "model_stats": "🤖 ماڊل ڪارڪردگي",
    "crop_input_header": "🌾 فصل ۽ موسم",
    "agri_input_header": "🧪 کاد ۽ ڪيڙي مار",
    "loading": "پيداوار اندازو حساب ٿي رهيو آهي...",
    "report_title": "فصل پيداوار اندازي جي رپورٽ",
    "report_download": "📥 رپورٽ ڊائونلوڊ ڪريو",
    "chart_soil": "مٽي غذائيت",
    "chart_trend": "پيداوار رجحان",
    "chart_weather": "موسم جائزو",
    "about_dev": "ٺاهيو",
    "about_model_title": "مشين لرننگ ماڊل",
    "about_features_title": "استعمال ٿيل خصوصيتون",
    "about_languages_title": "ٻوليون",
    "about_files_title": "فائلون",
}

TRANSLATIONS["कोंकणी (Konkani)"] = {
    "app_title": "🌾 पीक उत्पादन अंदाज",
    "app_subtitle": "भारतीय शेतकऱ्यांक लागून AI शेती बुद्धी",
    "language_label": "🌐 भास निवडात",
    "tab_predict": "🌱 उत्पादन अंदाज",
    "tab_recommend": "💡 पीक शिफारस",
    "tab_batch": "📋 बॅच अंदाज",
    "tab_about": "ℹ️ विशीं",
    "select_state": "राज्य निवडात",
    "select_crop": "पीक निवडात",
    "select_season": "हंगाम निवडात",
    "select_year": "वर्स निवडात",
    "fertilizer": "खत (kg/ha)",
    "pesticide": "कीटकनाशक (kg/ha)",
    "soil_header": "🪨 माती पॅरामीटर",
    "nitrogen": "नायट्रोजन (N) kg/ha",
    "phosphorus": "फॉस्फरस (P) kg/ha",
    "potassium": "पोटॅशियम (K) kg/ha",
    "soil_ph": "माती pH",
    "weather_header": "🌤️ हवामान पॅरामीटर",
    "temperature": "सरासरी तापमान (°C)",
    "rainfall": "वर्सुकी पावस (mm)",
    "humidity": "सरासरी आर्द्रता (%)",
    "lag_header": "📅 इतिहासीक डेटा (पर्यायी)",
    "yield_lag1": "फाटलो वर्स उत्पादन (kg/ha)",
    "yield_lag2": "2 वर्सा आदीं उत्पादन (kg/ha)",
    "predict_btn": "🔮 उत्पादन अंदाज करात",
    "predicted_yield": "अंदाजीत उत्पादन",
    "unit": "kg / हेक्टर",
    "model_not_found": "⚠️ मॉडेल फाइल मेळना। ML स्क्रिप्ट चलयात।",
    "result_header": "📊 अंदाजाचो निकाल",
    "good_yield": "✅ सरासरीपरस चड उत्पादन",
    "avg_yield": "⚠️ सरासरी उत्पादन",
    "low_yield": "❌ सरासरीपरस उणें उत्पादन",
    "tips_header": "💡 शेती टिप्स",
    "recommend_header": "🌿 प्रादेशिक पीक शिफारस",
    "recommend_state": "आपलें राज्य निवडात",
    "recommend_season": "हंगाम निवडात",
    "recommend_btn": "🔍 शिफारस मेळयात",
    "top_crops": "तुमच्या प्रदेशाक शिफारस केल्लीं पिकां",
    "batch_header": "📋 बॅच अंदाज",
    "upload_csv": "CSV फाइल अपलोड करात",
    "download_results": "⬇️ निकाल डाउनलोड करात",
    "about_header": "ह्या ॲपा विशीं",
    "model_stats": "🤖 मॉडेल कामगिरी",
    "crop_input_header": "🌾 पीक आनी हंगाम",
    "agri_input_header": "🧪 खत आनी कीटकनाशक",
    "loading": "उत्पादन अंदाजाची गणना जाता आसा...",
    "report_title": "पीक उत्पादन अंदाज रिपोर्ट",
    "report_download": "📥 रिपोर्ट डाउनलोड करात",
    "chart_soil": "माती पोषण वितरण",
    "chart_trend": "उत्पादन प्रवृत्ती",
    "chart_weather": "हवामान सारांश",
    "about_dev": "बनयलें",
    "about_model_title": "मशीन लर्निंग मॉडेल",
    "about_features_title": "वापरिल्लीं वैशिष्ट्यां",
    "about_languages_title": "आदारीत भासो",
    "about_files_title": "गरजेच्यो फाइली",
}

TRANSLATIONS["डोगरी (Dogri)"] = {
    "app_title": "🌾 फसल उत्पादन अनुमान",
    "app_subtitle": "भारतीय किसाना लेई AI कृषि बुद्धि",
    "language_label": "🌐 बोली चुनो",
    "tab_predict": "🌱 उत्पादन अनुमान",
    "tab_recommend": "💡 फसल सिफारश",
    "tab_batch": "📋 बैच अनुमान",
    "tab_about": "ℹ️ बारे च",
    "select_state": "सूबा चुनो",
    "select_crop": "फसल चुनो",
    "select_season": "मौसम चुनो",
    "select_year": "साल चुनो",
    "fertilizer": "खाद (kg/ha)",
    "pesticide": "कीड़ेमार (kg/ha)",
    "soil_header": "🪨 मिट्टी दे पैरामीटर",
    "nitrogen": "नाइट्रोजन (N) kg/ha",
    "phosphorus": "फास्फोरस (P) kg/ha",
    "potassium": "पोटाशियम (K) kg/ha",
    "soil_ph": "मिट्टी pH",
    "weather_header": "🌤️ मौसम दे पैरामीटर",
    "temperature": "औसत तापमान (°C)",
    "rainfall": "सालाना बरखा (mm)",
    "humidity": "औसत नमी (%)",
    "lag_header": "📅 पुराना डेटा (ऐच्छिक)",
    "yield_lag1": "पिछले साल दा उत्पादन (kg/ha)",
    "yield_lag2": "2 साल पैह्लें दा उत्पादन (kg/ha)",
    "predict_btn": "🔮 उत्पादन दा अनुमान लाओ",
    "predicted_yield": "अनुमानित उत्पादन",
    "unit": "kg / हेक्टेयर",
    "model_not_found": "⚠️ मॉडल फाइल नेईं मिली। ML स्क्रिप्ट चलाओ।",
    "result_header": "📊 अनुमान दा नतीजा",
    "good_yield": "✅ औसत थमां बेहतर उत्पादन",
    "avg_yield": "⚠️ औसत उत्पादन",
    "low_yield": "❌ औसत थमां घट्ट उत्पादन",
    "tips_header": "💡 खेती दे सुझाव",
    "recommend_header": "🌿 इलाकाई फसल सिफारश",
    "recommend_state": "अपना सूबा चुनो",
    "recommend_season": "मौसम चुनो",
    "recommend_btn": "🔍 सिफारशां लाओ",
    "top_crops": "तुंदे इलाके लेई फसलां",
    "batch_header": "📋 बैच अनुमान",
    "upload_csv": "CSV फाइल अपलोड करो",
    "download_results": "⬇️ नतीजे डाउनलोड करो",
    "about_header": "इस ऐप दे बारे च",
    "model_stats": "🤖 मॉडल प्रदर्शन",
    "crop_input_header": "🌾 फसल ते मौसम",
    "agri_input_header": "🧪 खाद ते कीड़ेमार",
    "loading": "उत्पादन अनुमान दी गणना होई रेई ऐ...",
    "report_title": "फसल उत्पादन अनुमान रिपोर्ट",
    "report_download": "📥 रिपोर्ट डाउनलोड करो",
    "chart_soil": "मिट्टी पोषण वितरण",
    "chart_trend": "उत्पादन रुझान",
    "chart_weather": "मौसम सारांश",
    "about_dev": "बनाया",
    "about_model_title": "मशीन लर्निंग मॉडल",
    "about_features_title": "इस्तेमाल विशेषताएं",
    "about_languages_title": "बोलियां",
    "about_files_title": "फाइलां",
}

TRANSLATIONS["बोड़ो (Bodo)"] = {
    "app_title": "🌾 फसल उत्पादन अनुमान",
    "app_subtitle": "भारतनि किसानफोरनि थाखाय AI बिथांमोन",
    "language_label": "🌐 बिथांखौ सायख",
    "tab_predict": "🌱 उत्पादन अनुमान",
    "tab_recommend": "💡 फसल सायख",
    "tab_batch": "📋 बैच अनुमान",
    "tab_about": "ℹ️ बिजाबाय",
    "select_state": "राज्यखौ सायख",
    "select_crop": "फसलखौ सायख",
    "select_season": "मौसमखौ सायख",
    "select_year": "बोसोरखौ सायख",
    "fertilizer": "खाद (kg/ha)",
    "pesticide": "कीटनाशक (kg/ha)",
    "soil_header": "🪨 माटिनि पारामिटार",
    "nitrogen": "नाइट्रोजन (N) kg/ha",
    "phosphorus": "फस्फोरस (P) kg/ha",
    "potassium": "पोटासियम (K) kg/ha",
    "soil_ph": "माटिनि pH",
    "weather_header": "🌤️ मौसमनि पारामिटार",
    "temperature": "मोगोन तापमान (°C)",
    "rainfall": "बोसोरनि बारिस (mm)",
    "humidity": "मोगोन आर्द्रता (%)",
    "lag_header": "📅 पुराना डेटा (ऐच्छिक)",
    "yield_lag1": "गोदां बोसोरनि उत्पादन (kg/ha)",
    "yield_lag2": "2 बोसोर सिगांनि उत्पादन (kg/ha)",
    "predict_btn": "🔮 उत्पादननि अनुमान लाग",
    "predicted_yield": "अनुमान उत्पादन",
    "unit": "kg / हेक्टर",
    "model_not_found": "⚠️ मॉडल फाइल मोनाय। ML स्क्रिप्ट थां।",
    "result_header": "📊 अनुमाननि फल",
    "good_yield": "✅ मोगोनथावनि जादो उत्पादन",
    "avg_yield": "⚠️ मोगोन उत्पादन",
    "low_yield": "❌ मोगोनथावनि कम उत्पादन",
    "tips_header": "💡 खेतिनि सुजु",
    "recommend_header": "🌿 इलाकानि फसल सायख",
    "recommend_state": "नांगौनि राज्यखौ सायख",
    "recommend_season": "मौसमखौ सायख",
    "recommend_btn": "🔍 सायखनाय लाग",
    "top_crops": "नांगौनि इलाकानि थाखाय फसल",
    "batch_header": "📋 बैच अनुमान",
    "upload_csv": "CSV फाइल अपलोड थां",
    "download_results": "⬇️ फल डाउनलोड थां",
    "about_header": "एबां एपनि बिजाबाय",
    "model_stats": "🤖 मॉडल परफर्मेन्स",
    "crop_input_header": "🌾 फसल आरो मौसम",
    "agri_input_header": "🧪 खाद आरो कीटनाशक",
    "loading": "उत्पादन अनुमान गुनाय जायो...",
    "report_title": "फसल उत्पादन अनुमान रिपोर्ट",
    "report_download": "📥 रिपोर्ट डाउनलोड थां",
    "chart_soil": "माटिनि पोषण",
    "chart_trend": "उत्पादन ट्रेन्ड",
    "chart_weather": "मौसम सारांश",
    "about_dev": "बानाय",
    "about_model_title": "मेशिन लर्निंग मॉडल",
    "about_features_title": "बावखां विशेषता",
    "about_languages_title": "बिथांफोर",
    "about_files_title": "फाइलफोर",
}

TRANSLATIONS["संताली (Santali)"] = {
    "app_title": "🌾 ᱯᱩᱛᱩᱞ ᱤᱫᱤ ᱚᱱᱩᱢᱟᱱ",
    "app_subtitle": "ᱵᱷᱟᱨᱚᱛ ᱟᱠᱷᱟᱱ ᱜᱮ AI ᱠᱷᱮᱛᱤ",
    "language_label": "🌐 ᱯᱷᱚᱱ ᱮᱱᱮᱢ",
    "tab_predict": "🌱 ᱤᱫᱤ ᱚᱱᱩᱢᱟᱱ",
    "tab_recommend": "💡 ᱯᱩᱛᱩᱞ ᱵᱟᱹᱨᱩ",
    "tab_batch": "📋 ᱵᱮᱪ ᱚᱱᱩᱢᱟᱱ",
    "tab_about": "ℹ️ ᱵᱤᱥᱚᱭ",
    "select_state": "ᱨᱟᱡᱽ ᱮᱱᱮᱢ",
    "select_crop": "ᱯᱩᱛᱩᱞ ᱮᱱᱮᱢ",
    "select_season": "ᱢᱚᱦᱩᱞᱟᱸ ᱮᱱᱮᱢ",
    "select_year": "ᱥᱮᱨᱢᱟ ᱮᱱᱮᱢ",
    "fertilizer": "ᱥᱟᱨ (kg/ha)",
    "pesticide": "ᱠᱤᱴᱱᱟᱥᱚᱠ (kg/ha)",
    "soil_header": "🪨 ᱢᱤᱴᱤ ᱯᱟᱨᱟᱢ",
    "nitrogen": "ᱱᱟᱭᱤᱴᱨᱚᱡᱮᱱ (N) kg/ha",
    "phosphorus": "ᱯᱷᱚᱥᱯᱷᱚᱨᱚᱥ (P) kg/ha",
    "potassium": "ᱯᱚᱴᱟᱥᱤᱭᱚᱢ (K) kg/ha",
    "soil_ph": "ᱢᱤᱴᱤ pH",
    "weather_header": "🌤️ ᱢᱚᱦᱩᱞᱟᱸ ᱯᱟᱨᱟᱢ",
    "temperature": "ᱟᱹᱪᱤᱡ ᱜᱟᱨᱢᱤ (°C)",
    "rainfall": "ᱥᱮᱨᱢᱟ ᱴᱮᱨ (mm)",
    "humidity": "ᱟᱹᱪᱤᱡ ᱱᱟᱢᱤ (%)",
    "lag_header": "📅 ᱯᱩᱨᱟᱱᱟ ᱰᱮᱴᱟ (ᱮᱪᱷᱤᱠ)",
    "yield_lag1": "ᱟᱜᱟᱜ ᱥᱮᱨᱢᱟ ᱤᱫᱤ (kg/ha)",
    "yield_lag2": "2 ᱥᱮᱨᱢᱟ ᱟᱜ ᱤᱫᱤ (kg/ha)",
    "predict_btn": "🔮 ᱤᱫᱤ ᱚᱱᱩᱢᱟᱱ",
    "predicted_yield": "ᱚᱱᱩᱢᱟᱱᱤᱛ ᱤᱫᱤ",
    "unit": "kg / ᱦᱮᱠᱴᱚᱨ",
    "model_not_found": "⚠️ ᱢᱚᱰᱮᱞ ᱯᱷᱟᱭᱞ ᱵᱟᱝ ᱢᱮᱱ। ML ᱪᱟᱞᱟᱣ।",
    "result_header": "📊 ᱚᱱᱩᱢᱟᱱ ᱯᱷᱞ",
    "good_yield": "✅ ᱟᱹᱪᱤᱡᱛᱮ ᱜᱮᱡ ᱤᱫᱤ",
    "avg_yield": "⚠️ ᱟᱹᱪᱤᱡ ᱤᱫᱤ",
    "low_yield": "❌ ᱟᱹᱪᱤᱡᱛᱮ ᱠᱚᱢ ᱤᱫᱤ",
    "tips_header": "💡 ᱠᱷᱮᱛᱤ ᱥᱩᱡᱟᱹ",
    "recommend_header": "🌿 ᱤᱞᱟᱠᱟ ᱯᱩᱛᱩᱞ ᱵᱟᱹᱨᱩ",
    "recommend_state": "ᱟᱯᱱᱟᱨ ᱨᱟᱡᱽ ᱮᱱᱮᱢ",
    "recommend_season": "ᱢᱚᱦᱩᱞᱟᱸ ᱮᱱᱮᱢ",
    "recommend_btn": "🔍 ᱵᱟᱹᱨᱩ ᱟᱹᱜᱩᱭ",
    "top_crops": "ᱟᱯᱱᱟᱨ ᱤᱞᱟᱠᱟ ᱯᱩᱛᱩᱞ",
    "batch_header": "📋 ᱵᱮᱪ ᱚᱱᱩᱢᱟᱱ",
    "upload_csv": "CSV ᱯᱷᱟᱭᱞ ᱟᱯᱞᱚᱰ",
    "download_results": "⬇️ ᱯᱷᱞ ᱰᱟᱣᱱᱞᱚᱰ",
    "about_header": "ᱱᱚᱶᱟ ᱮᱯ ᱵᱤᱥᱚᱭ",
    "model_stats": "🤖 ᱢᱚᱰᱮᱞ ᱯᱨᱚᱫᱚᱨᱥᱚᱱ",
    "crop_input_header": "🌾 ᱯᱩᱛᱩᱞ ᱟᱨ ᱢᱚᱦᱩᱞᱟᱸ",
    "agri_input_header": "🧪 ᱥᱟᱨ ᱟᱨ ᱠᱤᱴᱱᱟᱥᱚᱠ",
    "loading": "ᱤᱫᱤ ᱚᱱᱩᱢᱟᱱ ᱦᱤᱥᱟᱵ...",
    "report_title": "ᱯᱩᱛᱩᱞ ᱤᱫᱤ ᱚᱱᱩᱢᱟᱱ ᱨᱤᱯᱚᱨᱴ",
    "report_download": "📥 ᱨᱤᱯᱚᱨᱴ ᱰᱟᱣᱱᱞᱚᱰ",
    "chart_soil": "ᱢᱤᱴᱤ ᱯᱚᱥᱚᱱ",
    "chart_trend": "ᱤᱫᱤ ᱴᱨᱮᱱᱰ",
    "chart_weather": "ᱢᱚᱦᱩᱞᱟᱸ ᱥᱟᱨᱟᱸᱥ",
    "about_dev": "ᱵᱚᱱᱟᱣᱟᱜ",
    "about_model_title": "ᱢᱮᱥᱤᱱ ᱞᱮᱨᱱᱤᱝ ᱢᱚᱰᱮᱞ",
    "about_features_title": "ᱵᱮᱵᱷᱟᱨ ᱵᱤᱥᱮᱥᱛᱟ",
    "about_languages_title": "ᱯᱷᱚᱱᱯᱷᱚᱨ",
    "about_files_title": "ᱯᱷᱟᱭᱞᱯᱷᱚᱨ",
}


# ─────────────────────────────────────────────
#  REGIONAL CROP RECOMMENDATIONS DATABASE
# ─────────────────────────────────────────────
REGIONAL_CROPS = {
    "Punjab": {
        "Kharif": [
            {"crop": "Rice", "reason": "High water availability, ideal temperature 25-35°C", "avg_yield": "3800-4200 kg/ha", "icon": "🌾"},
            {"crop": "Maize", "reason": "Good drainage soil, warm climate", "avg_yield": "2500-3000 kg/ha", "icon": "🌽"},
            {"crop": "Cotton", "reason": "Sandy loam soil, long warm season", "avg_yield": "500-700 kg/ha", "icon": "☁️"},
            {"crop": "Groundnut", "reason": "Light sandy soil, moderate rainfall", "avg_yield": "1200-1500 kg/ha", "icon": "🥜"},
        ],
        "Rabi": [
            {"crop": "Wheat", "reason": "Alluvial soil, cool winter ideal", "avg_yield": "4000-5000 kg/ha", "icon": "🌿"},
            {"crop": "Mustard", "reason": "Well-drained soil, cool climate", "avg_yield": "1000-1200 kg/ha", "icon": "🌼"},
            {"crop": "Gram", "reason": "Light soil, low water requirement", "avg_yield": "800-1000 kg/ha", "icon": "🫘"},
        ],
    },
    "Maharashtra": {
        "Kharif": [
            {"crop": "Soyabean", "reason": "Black cotton soil, moderate rainfall", "avg_yield": "900-1200 kg/ha", "icon": "🌱"},
            {"crop": "Cotton", "reason": "Deep black soil, warm & dry", "avg_yield": "400-600 kg/ha", "icon": "☁️"},
            {"crop": "Sugarcane", "reason": "Heavy soil, high water availability", "avg_yield": "70000-90000 kg/ha", "icon": "🎋"},
            {"crop": "Rice", "reason": "Coastal regions, high rainfall", "avg_yield": "2000-2500 kg/ha", "icon": "🌾"},
        ],
        "Rabi": [
            {"crop": "Wheat", "reason": "Northern Maharashtra, fertile soil", "avg_yield": "2000-2500 kg/ha", "icon": "🌿"},
            {"crop": "Jowar", "reason": "Drought-resistant, dry climate", "avg_yield": "800-1000 kg/ha", "icon": "🌾"},
            {"crop": "Gram", "reason": "Well-drained soil, low rainfall", "avg_yield": "600-800 kg/ha", "icon": "🫘"},
        ],
    },
    "Tamil Nadu": {
        "Kharif": [
            {"crop": "Rice", "reason": "Delta regions, abundant water supply", "avg_yield": "2800-3500 kg/ha", "icon": "🌾"},
            {"crop": "Groundnut", "reason": "Red sandy loam soil", "avg_yield": "1000-1500 kg/ha", "icon": "🥜"},
            {"crop": "Cotton", "reason": "Coimbatore region, black soil", "avg_yield": "400-500 kg/ha", "icon": "☁️"},
        ],
        "Rabi": [
            {"crop": "Rice", "reason": "Cauvery delta, year-round cultivation", "avg_yield": "2500-3000 kg/ha", "icon": "🌾"},
            {"crop": "Sugarcane", "reason": "Irrigation facilities, warm climate", "avg_yield": "80000-100000 kg/ha", "icon": "🎋"},
            {"crop": "Banana", "reason": "Tropical climate, fertile soil", "avg_yield": "25000-35000 kg/ha", "icon": "🍌"},
        ],
    },
    "Uttar Pradesh": {
        "Kharif": [
            {"crop": "Rice", "reason": "Eastern UP, high rainfall & alluvial soil", "avg_yield": "2200-2800 kg/ha", "icon": "🌾"},
            {"crop": "Sugarcane", "reason": "Western UP, Ganga-Yamuna doab", "avg_yield": "65000-80000 kg/ha", "icon": "🎋"},
            {"crop": "Maize", "reason": "Hilly terrains, moderate climate", "avg_yield": "2000-2500 kg/ha", "icon": "🌽"},
        ],
        "Rabi": [
            {"crop": "Wheat", "reason": "Most cultivated crop, fertile alluvial soil", "avg_yield": "3000-3800 kg/ha", "icon": "🌿"},
            {"crop": "Mustard", "reason": "Semi-arid regions, well drained soil", "avg_yield": "900-1100 kg/ha", "icon": "🌼"},
            {"crop": "Potato", "reason": "Cool climate, sandy loam soil", "avg_yield": "20000-25000 kg/ha", "icon": "🥔"},
        ],
    },
    "Rajasthan": {
        "Kharif": [
            {"crop": "Bajra", "reason": "Sandy soil, drought-tolerant", "avg_yield": "800-1200 kg/ha", "icon": "🌾"},
            {"crop": "Groundnut", "reason": "Sandy loam, dry climate", "avg_yield": "800-1000 kg/ha", "icon": "🥜"},
            {"crop": "Cotton", "reason": "Irrigated areas, warm dry climate", "avg_yield": "350-500 kg/ha", "icon": "☁️"},
        ],
        "Rabi": [
            {"crop": "Wheat", "reason": "Irrigated plains, fertile soil", "avg_yield": "2500-3000 kg/ha", "icon": "🌿"},
            {"crop": "Mustard", "reason": "Largest mustard producer, semi-arid soil", "avg_yield": "1200-1500 kg/ha", "icon": "🌼"},
            {"crop": "Gram", "reason": "Dry cold climate, light soil", "avg_yield": "700-900 kg/ha", "icon": "🫘"},
        ],
    },
    "West Bengal": {
        "Kharif": [
            {"crop": "Rice", "reason": "Alluvial delta soil, high rainfall", "avg_yield": "2500-3200 kg/ha", "icon": "🌾"},
            {"crop": "Jute", "reason": "Humid delta, highest jute producer", "avg_yield": "2000-2500 kg/ha", "icon": "🌿"},
            {"crop": "Maize", "reason": "Highland areas, sandy loam soil", "avg_yield": "1800-2200 kg/ha", "icon": "🌽"},
        ],
        "Rabi": [
            {"crop": "Wheat", "reason": "Northern plains, cool dry climate", "avg_yield": "2000-2500 kg/ha", "icon": "🌿"},
            {"crop": "Mustard", "reason": "Gangetic plains, fertile soil", "avg_yield": "800-1000 kg/ha", "icon": "🌼"},
            {"crop": "Potato", "reason": "Cool climate, Hooghly district", "avg_yield": "18000-22000 kg/ha", "icon": "🥔"},
        ],
    },
    "Kerala": {
        "Kharif": [
            {"crop": "Rice", "reason": "Coastal plains, kuttanad wetlands", "avg_yield": "2000-2500 kg/ha", "icon": "🌾"},
            {"crop": "Coconut", "reason": "Coastal area, tropical climate", "avg_yield": "8000-12000 kg/ha", "icon": "🥥"},
            {"crop": "Banana", "reason": "Humid tropical, year-round", "avg_yield": "20000-30000 kg/ha", "icon": "🍌"},
        ],
        "Rabi": [
            {"crop": "Tapioca", "reason": "Laterite soil, moderate rainfall", "avg_yield": "25000-35000 kg/ha", "icon": "🌿"},
            {"crop": "Pepper", "reason": "Hilly regions, spice capital", "avg_yield": "300-500 kg/ha", "icon": "🌶️"},
        ],
    },
    "Andhra Pradesh": {
        "Kharif": [
            {"crop": "Rice", "reason": "Krishna-Godavari delta, high fertility", "avg_yield": "3000-4000 kg/ha", "icon": "🌾"},
            {"crop": "Cotton", "reason": "Black cotton soil, Guntur region", "avg_yield": "450-600 kg/ha", "icon": "☁️"},
            {"crop": "Groundnut", "reason": "Red loamy soil, Rayalaseema", "avg_yield": "1000-1400 kg/ha", "icon": "🥜"},
        ],
        "Rabi": [
            {"crop": "Maize", "reason": "Fertile soil, good rainfall", "avg_yield": "3000-4000 kg/ha", "icon": "🌽"},
            {"crop": "Sugarcane", "reason": "Irrigated areas, warm climate", "avg_yield": "75000-90000 kg/ha", "icon": "🎋"},
        ],
    },
}

# Default crops for states not in the database
DEFAULT_CROPS = {
    "Kharif": [
        {"crop": "Rice", "reason": "Staple Kharif crop for most Indian states", "avg_yield": "2000-3000 kg/ha", "icon": "🌾"},
        {"crop": "Maize", "reason": "Versatile crop, suits most soils", "avg_yield": "2000-2500 kg/ha", "icon": "🌽"},
        {"crop": "Groundnut", "reason": "Cash crop, moderate rainfall needed", "avg_yield": "1000-1500 kg/ha", "icon": "🥜"},
    ],
    "Rabi": [
        {"crop": "Wheat", "reason": "Major Rabi crop, alluvial soil", "avg_yield": "2500-3500 kg/ha", "icon": "🌿"},
        {"crop": "Mustard", "reason": "Oilseed, cool dry climate", "avg_yield": "800-1200 kg/ha", "icon": "🌼"},
        {"crop": "Gram", "reason": "Legume, low water requirement", "avg_yield": "700-1000 kg/ha", "icon": "🫘"},
    ],
    "Whole Year": [
        {"crop": "Sugarcane", "reason": "Year-round crop, irrigated areas", "avg_yield": "60000-80000 kg/ha", "icon": "🎋"},
        {"crop": "Banana", "reason": "Tropical perennial", "avg_yield": "20000-30000 kg/ha", "icon": "🍌"},
        {"crop": "Coconut", "reason": "Coastal/tropical perennial", "avg_yield": "8000-12000 kg/ha", "icon": "🥥"},
    ],
}

# ─────────────────────────────────────────────
#  FEATURES LIST (must match trained model)
# ─────────────────────────────────────────────
FEATURES = [
    'state_enc', 'crop_enc', 'season_enc', 'year',
    'log_fertilizer', 'log_pesticide',
    'n', 'p', 'k', 'ph', 'npk_total', 'npk_ratio_np',
    'rain_temp_interact', 'fert_per_rain',
    'avg_temp_c', 'total_rainfall_mm', 'avg_humidity_percent',
    'yield_diff', 'rainfall_diff', 'temp_diff',
    'yield_lag1_log', 'yield_lag2_log', 'yield_ma2_log', 'rain_lag1'
]

# ─────────────────────────────────────────────
#  CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans', sans-serif; }

/* BACKGROUND */
.stApp { background: linear-gradient(160deg, #0d2b08 0%, #1a4a0a 40%, #0a3d15 100%) !important; }
.main .block-container { padding: 1.5rem 2rem; max-width: 1280px; }

/* HEADER */
.app-header {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px; padding: 2rem 2.5rem; color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    position: relative; overflow: hidden;
}
.app-header h1 { color: white; margin: 0; font-size: 2rem; font-weight: 800; }
.app-header p  { color: rgba(255,255,255,0.8); margin: 0.3rem 0 0 0; font-size: 1rem; }

/* CROP PILLS */
.crop-pill {
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3);
    border-radius: 50px; padding: 4px 14px; font-size: 0.78rem; color: white; font-weight: 600;
    display: inline-block;
}

/* HEADER STATS */
.header-stat {
    background: rgba(0,0,0,0.25); border-radius: 14px; padding: 10px 20px;
    min-width: 90px; text-align: center; border: 1px solid rgba(255,255,255,0.15); display: inline-block;
}
.header-stat-num   { font-size: 1.6rem; font-weight: 900; color: #CCFF90; line-height: 1; }
.header-stat-label { font-size: 0.62rem; color: rgba(255,255,255,0.65); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 3px; }

/* TABS */
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(255,255,255,0.97); border-radius: 0 16px 16px 16px;
    padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 14px 14px 0 0 !important; padding: 6px 6px 0 6px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.12) !important; color: rgba(255,255,255,0.85) !important;
    border-radius: 10px 10px 0 0 !important; font-weight: 600 !important; font-size: 0.88rem !important;
    padding: 8px 18px !important; border: none !important;
}
.stTabs [aria-selected="true"] { background: white !important; color: #1a4a0a !important; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,0.25) !important; }

/* INPUTS */
.stSelectbox > label, .stNumberInput > label, .stSlider > label {
    color: #1a4a0a !important; font-weight: 700 !important;
    font-size: 0.82rem !important; text-transform: uppercase !important; letter-spacing: 0.5px !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #f0f7e8 !important; border: 2px solid #c8e6c9 !important;
    border-radius: 10px !important; color: #1a4a0a !important; font-weight: 600 !important;
}
.stNumberInput input {
    background: #f0f7e8 !important; border: 2px solid #c8e6c9 !important;
    border-radius: 10px !important; color: #1a4a0a !important; font-weight: 600 !important;
}

/* SECTION TITLES */
.section-title {
    color: #1a4a0a; font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 2.5px; padding: 0.4rem 0; border-bottom: 2px solid #4CAF50;
    margin: 1.4rem 0 1rem 0;
}

/* PREDICT BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #1a4a0a, #2d7a14, #43A047) !important;
    color: white !important; border: none !important; border-radius: 14px !important;
    font-weight: 800 !important; font-size: 1rem !important; padding: 0.8rem 2rem !important;
    width: 100% !important; box-shadow: 0 6px 20px rgba(45,122,20,0.4) !important;
    transition: all 0.2s ease !important; text-transform: uppercase !important;
}
.stButton > button:hover { transform: translateY(-3px) !important; box-shadow: 0 10px 28px rgba(45,122,20,0.55) !important; }

/* RESULT CARD */
.result-card {
    background: linear-gradient(135deg, #1a4a0a 0%, #2d7a14 60%, #43A047 100%);
    border-radius: 20px; padding: 2rem; color: white; text-align: center;
    margin: 0.8rem 0; box-shadow: 0 10px 36px rgba(45,122,20,0.4);
}
.yield-value { font-size: 3.8rem; font-weight: 900; letter-spacing: -2px; line-height: 1; }
.yield-label { font-size: 0.8rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 0.5rem; }
.yield-unit  { font-size: 1rem; opacity: 0.75; margin-top: 0.4rem; }
.yield-badge { display: inline-block; background: rgba(255,255,255,0.22); border-radius: 50px; padding: 0.3rem 1.2rem; margin-top: 1rem; font-size: 0.9rem; font-weight: 700; border: 1px solid rgba(255,255,255,0.3); }

/* STAT CHIPS */
.stat-row { display: flex; gap: 0.6rem; margin: 0.8rem 0; flex-wrap: wrap; }
.stat-chip {
    background: #e8f5e9; border-radius: 50px; padding: 0.4rem 1rem;
    font-size: 0.82rem; font-weight: 700; color: #1a4a0a; border: 1.5px solid #c8e6c9;
}

/* CROP CARDS */
.crop-card {
    background: white; border-radius: 14px; padding: 1.2rem; margin: 0.6rem 0;
    border-left: 5px solid #4CAF50; box-shadow: 0 3px 14px rgba(0,0,0,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
}
.crop-card:hover { transform: translateX(5px); box-shadow: 0 5px 18px rgba(0,0,0,0.12); }
.crop-card .crop-name   { font-weight: 800; font-size: 1.05rem; color: #1a4a0a; }
.crop-card .crop-reason { font-size: 0.85rem; color: #555; margin-top: 0.3rem; }
.crop-card .crop-yield  { font-size: 0.85rem; color: #2d7a14; font-weight: 700; margin-top: 0.4rem; }

/* CHART CARDS */
.chart-card {
    background: white;
    border-radius: 14px;
    padding: 0.8rem;
    box-shadow: 0 3px 14px rgba(0,0,0,0.08);
    border: 1px solid #e8f5e9;
    margin: 0.4rem 0;
}
.chart-title {
    font-size: 0.75rem;
    font-weight: 800;
    color: #1a4a0a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

/* INFO BOX */
.info-box {
    background: #e8f5e9; border-radius: 10px; padding: 0.9rem 1.2rem;
    border: 1px solid #a5d6a7; margin: 0.6rem 0; font-size: 0.88rem;
    color: #1b5e20; font-weight: 500;
}

/* ABOUT CARDS */
.about-card {
    background: white; border-radius: 18px; padding: 1.5rem;
    box-shadow: 0 5px 22px rgba(0,0,0,0.1); border-top: 5px solid #4CAF50; margin-bottom: 1rem;
}
.about-card-icon  { font-size: 2.2rem; margin-bottom: 0.6rem; }
.about-card-title { font-size: 0.9rem; font-weight: 800; color: #1a4a0a; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 1.2px; }
.about-card-body  { font-size: 0.87rem; color: #333; line-height: 2; }
.about-card-body code { background: #f0f7e8; padding: 2px 6px; border-radius: 5px; font-size: 0.8rem; color: #1a4a0a; }

/* SIDEBAR */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #071a04 0%, #0d2b08 100%) !important; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: #c8e6c9 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #CCFF90 !important; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.1) !important; border: 1.5px solid rgba(255,255,255,0.2) !important;
    color: white !important; border-radius: 10px !important;
}

/* DOWNLOAD BUTTON */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1565C0, #1976D2) !important;
    color: white !important; border-radius: 12px !important; font-weight: 700 !important;
    border: none !important; box-shadow: 0 4px 16px rgba(21,101,192,0.35) !important;
}
.stDownloadButton > button:hover { transform: translateY(-2px) !important; }

/* METRICS */
[data-testid="metric-container"] {
    background: white; border-radius: 14px; padding: 1rem !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08); border-bottom: 3px solid #4CAF50;
}
[data-testid="stMetricValue"] { color: #1a4a0a !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #555 !important; }

/* EXPANDER */
.streamlit-expanderHeader {
    background: #f0f7e8 !important; border-radius: 10px !important;
    color: #1a4a0a !important; font-weight: 700 !important; border: 1.5px solid #c8e6c9 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] { background: #f0f7e8 !important; border: 2px dashed #4CAF50 !important; border-radius: 14px !important; }

/* FOOTER */
.footer {
    text-align: center; padding: 1.5rem; margin-top: 2rem;
    border-top: 2px solid rgba(255,255,255,0.12); color: rgba(255,255,255,0.6);
    font-size: 0.88rem; background: rgba(255,255,255,0.05); border-radius: 14px;
}
.footer b { color: #CCFF90; font-size: 1rem; }

/* PROGRESS + DATAFRAME */
.stProgress > div > div { background: #4CAF50 !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path="crop_yield_model.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    return None

bundle = load_model()

# ─────────────────────────────────────────────
#  PREDICT FUNCTION
# ─────────────────────────────────────────────
def predict(bundle, state, crop, season, year,
            fertilizer, pesticide, n, p, k, ph,
            avg_temp_c, total_rainfall_mm, avg_humidity_percent,
            yield_lag1=0.0, yield_lag2=0.0, rain_lag1=0.0,
            yield_diff=0.0, rainfall_diff=0.0, temp_diff=0.0):
    import numpy as np
    import pandas as pd
    model = bundle['model']
    le_state = bundle['le_state']
    le_crop = bundle['le_crop']
    le_season = bundle['le_season']

    for name, le, val in [('State', le_state, state),
                           ('Crop', le_crop, crop),
                           ('Season', le_season, season)]:
        if val not in le.classes_:
            raise ValueError(f"{name} '{val}' not recognized. Available: {sorted(le.classes_.tolist())}")

    npk_total = n + p + k
    npk_ratio_np = n / (p + 1)
    rain_temp_interact = total_rainfall_mm * avg_temp_c
    fert_per_rain = fertilizer / (total_rainfall_mm + 1)
    yield_lag1_log = np.log1p(yield_lag1)
    yield_lag2_log = np.log1p(yield_lag2)
    yield_ma2_log = (yield_lag1_log + yield_lag2_log) / 2
    log_fertilizer = np.log1p(fertilizer)
    log_pesticide = np.log1p(pesticide)

    row = pd.DataFrame([[
        le_state.transform([state])[0],
        le_crop.transform([crop])[0],
        le_season.transform([season])[0],
        year, log_fertilizer, log_pesticide,
        n, p, k, ph, npk_total, npk_ratio_np,
        rain_temp_interact, fert_per_rain,
        avg_temp_c, total_rainfall_mm, avg_humidity_percent,
        yield_diff, rainfall_diff, temp_diff,
        yield_lag1_log, yield_lag2_log, yield_ma2_log, rain_lag1
    ]], columns=FEATURES)

    pred_log = model.predict(row)[0]
    return float(np.expm1(pred_log))

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'English'

# ─────────────────────────────────────────────
#  SIDEBAR — LANGUAGE SELECTOR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌐 Language / भाषा")
    selected_lang = st.selectbox(
        "Choose Language",
        list(TRANSLATIONS.keys()),
        index=list(TRANSLATIONS.keys()).index(st.session_state['lang']),
        label_visibility="collapsed"
    )
    st.session_state['lang'] = selected_lang

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8rem;color:#1a4a0a;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
        📊 Model Info
    </div>
    """, unsafe_allow_html=True)
    st.markdown("📈 **R² Score:** 95.73%")
    st.markdown("🌿 **Algorithm:** Gradient Boosting")
    st.markdown("📅 **Data:** 1997–2020")
    st.markdown("🗺️ **States:** 29 Indian States")
    st.markdown("🌐 **Languages:** 22 (All 8th Schedule)")
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#888;text-align:center;">
        🌾 Agri AI<br>
        Developed by <b style="color:#2d7a14;">Ritvik</b>
    </div>
    """, unsafe_allow_html=True)

T = TRANSLATIONS[selected_lang]

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:1rem;">
        <span class="crop-pill">🌾 Wheat</span>
        <span class="crop-pill">🌾 Rice</span>
        <span class="crop-pill">🌽 Maize</span>
        <span class="crop-pill">🌱 Soyabean</span>
        <span class="crop-pill">🎋 Sugarcane</span>
        <span class="crop-pill">🌻 Sunflower</span>
        <span class="crop-pill">🥔 Potato</span>
        <span class="crop-pill">+ 50 more</span>
    </div>
    <h1>{T['app_title']}</h1>
    <p>{T['app_subtitle']}</p>
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:1.2rem;">
        <div class="header-stat"><div class="header-stat-num">95.7%</div><div class="header-stat-label">MODEL ACCURACY</div></div>
        <div class="header-stat"><div class="header-stat-num">29</div><div class="header-stat-label">INDIAN STATES</div></div>
        <div class="header-stat"><div class="header-stat-num">23 yrs</div><div class="header-stat-label">TRAINING DATA</div></div>
        <div class="header-stat"><div class="header-stat-num">22</div><div class="header-stat-label">LANGUAGES</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    T["tab_predict"],
    T["tab_recommend"],
    T["tab_batch"],
    T["tab_about"]
])

# ══════════════════════════════════════════════
#  TAB 1 — PREDICT YIELD
# ══════════════════════════════════════════════
with tab1:
    if bundle is None:
        st.error(T["model_not_found"])
        st.info("📌 Run the training script first:\n```bash\npython crop_yield_model.py\n```")
    else:
        states  = bundle['states']
        crops   = bundle['crops']
        seasons = bundle['seasons']

        col_left, col_right = st.columns([1.1, 0.9], gap="large")

        with col_left:
            # ── Crop & Season ──────────────────────────
            st.markdown(f'<div class="section-title">{T["crop_input_header"]}</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                state  = st.selectbox(T["select_state"],  states,  key="state")
            with c2:
                crop   = st.selectbox(T["select_crop"],   crops,   key="crop")
            c3, c4 = st.columns(2)
            with c3:
                season = st.selectbox(T["select_season"], seasons, key="season")
            with c4:
                year   = st.slider(T["select_year"], 1997, 2025, 2020, key="year")

            # ── Fertilizer & Pesticide ─────────────────
            st.markdown(f'<div class="section-title">{T["agri_input_header"]}</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                fertilizer = st.number_input(T["fertilizer"], 0.0, 1000.0, 150.0, step=5.0)
            with c6:
                pesticide  = st.number_input(T["pesticide"],  0.0,  50.0,   2.5,  step=0.1)

            # ── Soil ───────────────────────────────────
            st.markdown(f'<div class="section-title">{T["soil_header"]}</div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            with s1: n  = st.number_input(T["nitrogen"],   0.0, 200.0, 80.0)
            with s2: p  = st.number_input(T["phosphorus"], 0.0, 200.0, 40.0)
            with s3: k  = st.number_input(T["potassium"],  0.0, 200.0, 40.0)
            with s4: ph = st.number_input(T["soil_ph"],    3.0,  10.0,  7.0, step=0.1)

            # ── Weather ────────────────────────────────
            st.markdown(f'<div class="section-title">{T["weather_header"]}</div>', unsafe_allow_html=True)
            w1, w2, w3 = st.columns(3)
            with w1: temp     = st.number_input(T["temperature"], 5.0,  50.0, 27.5, step=0.5)
            with w2: rainfall = st.number_input(T["rainfall"],    50.0, 4000.0, 850.0, step=10.0)
            with w3: humidity = st.number_input(T["humidity"],    10.0, 100.0,  65.0, step=1.0)

            # ── Historical ─────────────────────────────
            with st.expander(T["lag_header"]):
                h1, h2 = st.columns(2)
                with h1: lag1 = st.number_input(T["yield_lag1"], 0.0, 20000.0, 0.0)
                with h2: lag2 = st.number_input(T["yield_lag2"], 0.0, 20000.0, 0.0)

            # ── PREDICT BUTTON ─────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            predict_clicked = st.button(T["predict_btn"], use_container_width=True)

        with col_right:
            st.markdown(f'<div class="section-title">{T["result_header"]}</div>', unsafe_allow_html=True)

            if predict_clicked:
                with st.spinner(T["loading"]):
                    try:
                        result = predict(
                            bundle, state, crop, season, year,
                            fertilizer, pesticide,
                            n, p, k, ph,
                            temp, rainfall, humidity,
                            yield_lag1=lag1, yield_lag2=lag2
                        )

                        # Determine performance badge
                        if result >= 3000:
                            badge = T["good_yield"]
                            badge_color = "#4CAF50"
                        elif result >= 1500:
                            badge = T["avg_yield"]
                            badge_color = "#FF9800"
                        else:
                            badge = T["low_yield"]
                            badge_color = "#f44336"

                        st.markdown(f"""
                        <div class="result-card">
                            <div class="yield-label">{T['predicted_yield']}</div>
                            <div class="yield-value">{result:,.0f}</div>
                            <div class="yield-unit">{T['unit']}</div>
                            <div class="yield-badge">{badge}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Input summary chips
                        st.markdown(f"""
                        <div class="stat-row">
                            <span class="stat-chip">🌍 {state}</span>
                            <span class="stat-chip">🌾 {crop}</span>
                            <span class="stat-chip">📅 {season} {year}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-chip">🌡️ {temp}°C</span>
                            <span class="stat-chip">🌧️ {rainfall}mm</span>
                            <span class="stat-chip">💧 {humidity}%</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Farming tips
                        st.markdown(f'<div class="section-title">{T["tips_header"]}</div>', unsafe_allow_html=True)
                        tips = []
                        if n < 50:  tips.append("⚠️ Low Nitrogen — consider urea or DAP application")
                        if p < 25:  tips.append("⚠️ Low Phosphorus — apply SSP or DAP")
                        if k < 25:  tips.append("⚠️ Low Potassium — apply MOP (muriate of potash)")
                        if ph < 6:  tips.append("🪨 Acidic soil — apply lime to raise pH")
                        if ph > 8:  tips.append("🪨 Alkaline soil — apply gypsum or sulphur")
                        if rainfall < 500: tips.append("💧 Low rainfall — ensure irrigation is available")
                        if fertilizer < 50: tips.append("🌱 Low fertilizer — increase dosage for better yield")
                        if not tips: tips = ["✅ Soil and weather conditions look good!", "🌾 Ensure timely sowing for best results."]

                        for tip in tips:
                            st.markdown(f'<div class="info-box">{tip}</div>', unsafe_allow_html=True)

                        # Charts - pure HTML/SVG, zero external libraries
                        st.markdown(f'<div class="section-title">📊 {T.get("chart_soil", "Soil Nutrient Distribution")}</div>', unsafe_allow_html=True)
                        chart_c1, chart_c2 = st.columns(2)

                        with chart_c1:
                            import math as _math
                            total_npk = max(n + p + k, 0.1)
                            pct_n = max(n, 0) / total_npk * 100
                            pct_p = max(p, 0) / total_npk * 100
                            pct_k = max(k, 0) / total_npk * 100
                            def _arc(cx, cy, r, s_deg, e_deg, color):
                                if e_deg - s_deg < 1: return ""
                                s = _math.radians(s_deg - 90)
                                e = _math.radians(e_deg - 90)
                                ri = r * 0.55
                                large = 1 if (e_deg - s_deg) > 180 else 0
                                x1,y1 = cx+r*_math.cos(s), cy+r*_math.sin(s)
                                x2,y2 = cx+r*_math.cos(e), cy+r*_math.sin(e)
                                xi1,yi1 = cx+ri*_math.cos(e), cy+ri*_math.sin(e)
                                xi2,yi2 = cx+ri*_math.cos(s), cy+ri*_math.sin(s)
                                return f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f} L {xi1:.1f} {yi1:.1f} A {ri} {ri} 0 {large} 0 {xi2:.1f} {yi2:.1f} Z" fill="{color}" stroke="white" stroke-width="2"/>'
                            a1,a2,a3 = 0, pct_n*3.6, pct_n*3.6+pct_p*3.6
                            donut_svg = f"""<svg width="150" height="150" viewBox="0 0 150 150">
                              {_arc(75,75,65,a1,a2,'#2d7a14')}
                              {_arc(75,75,65,a2,a3,'#66BB6A')}
                              {_arc(75,75,65,a3,360,'#C8E6C9')}
                              <circle cx="75" cy="75" r="36" fill="#f0f7e8"/>
                              <text x="75" y="71" text-anchor="middle" font-size="10" font-weight="bold" fill="#1a4a0a">N+P+K</text>
                              <text x="75" y="86" text-anchor="middle" font-size="13" font-weight="bold" fill="#2d7a14">{int(n+p+k)}</text>
                            </svg>"""
                            st.markdown(f"""<div class="chart-card" style="text-align:center;padding:1rem;">
                              <div class="chart-title">{T.get('chart_soil','Soil Nutrients')}</div>
                              {donut_svg}
                              <div style="display:flex;justify-content:center;gap:14px;margin-top:8px;flex-wrap:wrap;">
                                <span style="font-size:0.78rem;"><b style="color:#2d7a14;">N</b> {pct_n:.0f}%</span>
                                <span style="font-size:0.78rem;"><b style="color:#66BB6A;">P</b> {pct_p:.0f}%</span>
                                <span style="font-size:0.78rem;"><b style="color:#A5D6A7;">K</b> {pct_k:.0f}%</span>
                              </div>
                            </div>""", unsafe_allow_html=True)

                        with chart_c2:
                            rain_range = list(range(max(50, int(rainfall)-300), int(rainfall)+400, 50))
                            sim_yields = [round(result * (0.6 + 0.8 * (r / (rainfall + 1))**0.5), 0) for r in rain_range]
                            mn, mx = min(sim_yields), max(sim_yields)
                            y_span = max(mx - mn, 1)
                            W, H, PX, PY = 240, 140, 24, 16
                            def _sx(r): return PX + (r - rain_range[0]) / max(rain_range[-1]-rain_range[0],1) * (W-2*PX)
                            def _sy(y): return H - PY - ((y - mn) / y_span) * (H - 2*PY)
                            pts = " ".join(f"{_sx(r):.1f},{_sy(y):.1f}" for r,y in zip(rain_range, sim_yields))
                            fill_pts = f"{_sx(rain_range[0]):.1f},{H-PY} {pts} {_sx(rain_range[-1]):.1f},{H-PY}"
                            vx = _sx(rainfall)
                            label_x = min(vx + 4, W - 55)
                            line_svg = f"""<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="overflow:visible">
                              <polygon points="{fill_pts}" fill="#4CAF50" opacity="0.15"/>
                              <polyline points="{pts}" fill="none" stroke="#2d7a14" stroke-width="2.5" stroke-linejoin="round"/>
                              <line x1="{vx:.1f}" y1="{PY}" x2="{vx:.1f}" y2="{H-PY}" stroke="#f44336" stroke-width="2" stroke-dasharray="5,3"/>
                              <text x="{label_x:.1f}" y="{PY+10}" font-size="9" fill="#f44336" font-weight="bold">You: {rainfall:.0f}mm</text>
                              <text x="{PX}" y="{H-2}" font-size="8" fill="#888">{rain_range[0]}mm</text>
                              <text x="{W-PX-18}" y="{H-2}" font-size="8" fill="#888">{rain_range[-1]}mm</text>
                            </svg>"""
                            st.markdown(f"""<div class="chart-card" style="text-align:center;padding:1rem;">
                              <div class="chart-title">{T.get('chart_trend','Yield vs Rainfall Trend')}</div>
                              {line_svg}
                              <div style="font-size:0.75rem;color:#888;margin-top:4px;">Rainfall (mm) vs Estimated Yield (kg/ha)</div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown(f'<div class="section-title">🌤️ {T.get("chart_weather", "Weather Overview")}</div>', unsafe_allow_html=True)
                        w_items = [
                            ("🌡️ Temperature", temp, f"{temp}°C", "#FF7043"),
                            ("🌧️ Rainfall (÷10)", rainfall/10, f"{rainfall:.0f}mm", "#42A5F5"),
                            ("💧 Humidity", humidity, f"{humidity}%", "#66BB6A"),
                        ]
                        w_max = max(v for _,v,_,_ in w_items) * 1.1
                        bars_html = ""
                        for lbl, val, disp, col in w_items:
                            pct = val / w_max * 100
                            bars_html += f"""<div style="margin:10px 0;display:flex;align-items:center;gap:10px;">
                              <div style="width:120px;font-size:0.82rem;font-weight:600;color:#333;text-align:right;flex-shrink:0;">{lbl}</div>
                              <div style="flex:1;background:#e8f5e9;border-radius:8px;height:26px;position:relative;overflow:hidden;">
                                <div style="width:{pct:.1f}%;background:{col};height:100%;border-radius:8px;"></div>
                              </div>
                              <div style="width:55px;font-size:0.85rem;font-weight:800;color:#1a4a0a;flex-shrink:0;">{disp}</div>
                            </div>"""
                        st.markdown(f'<div class="chart-card" style="padding:1rem 1.2rem;">{bars_html}</div>', unsafe_allow_html=True)


                        # ── PDF REPORT DOWNLOAD ────────────────────
                        st.markdown(f'<div class="section-title">📥 {T.get("report_download", "Download Report")}</div>', unsafe_allow_html=True)

                        badge_text = badge.replace("✅ ","").replace("⚠️ ","").replace("❌ ","")
                        report_lang = selected_lang

                        report_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap');
  body {{ font-family: 'Noto Sans', sans-serif; margin: 40px; color: #222; background: #f9fdf4; }}
  h1 {{ color: #1a4a0a; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
  h2 {{ color: #2d7a14; margin-top: 25px; }}
  .badge {{ display:inline-block; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1em;
            background: {'#e8f5e9' if result >= 3000 else '#fff3e0' if result >= 1500 else '#fce4ec'};
            color: {'#1b5e20' if result >= 3000 else '#e65100' if result >= 1500 else '#b71c1c'}; }}
  .result-box {{ background: linear-gradient(135deg,#1a4a0a,#2d7a14); color:white; padding:25px;
                 border-radius:16px; text-align:center; margin:20px 0; }}
  .result-box .num {{ font-size:3em; font-weight:900; }}
  .result-box .unit {{ font-size:1em; opacity:0.8; }}
  table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
  th {{ background:#2d7a14; color:white; padding:8px 12px; text-align:left; }}
  td {{ padding:8px 12px; border-bottom:1px solid #ddd; }}
  tr:nth-child(even) {{ background:#f0f7e8; }}
  .tip {{ background:#e8f5e9; border-left:4px solid #4CAF50; padding:10px 14px; margin:8px 0; border-radius:6px; }}
  footer {{ text-align:center; margin-top:40px; color:#888; font-size:0.85em; border-top:1px solid #ccc; padding-top:12px; }}
</style>
</head>
<body>
<h1>🌾 {T.get('report_title', 'Crop Yield Prediction Report')}</h1>
<p><b>Language / भाषा:</b> {report_lang}</p>

<div class="result-box">
  <div style="opacity:0.8;font-size:0.85em;text-transform:uppercase;letter-spacing:3px;">{T['predicted_yield']}</div>
  <div class="num">{result:,.0f}</div>
  <div class="unit">{T['unit']}</div>
</div>
<div style="text-align:center;margin:12px 0;"><span class="badge">{badge}</span></div>

<h2>📋 Input Parameters</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>🌍 State</td><td>{state}</td></tr>
  <tr><td>🌾 Crop</td><td>{crop}</td></tr>
  <tr><td>📅 Season</td><td>{season}</td></tr>
  <tr><td>🗓️ Year</td><td>{year}</td></tr>
  <tr><td>🧪 Fertilizer</td><td>{fertilizer} kg/ha</td></tr>
  <tr><td>🐛 Pesticide</td><td>{pesticide} kg/ha</td></tr>
  <tr><td>N (Nitrogen)</td><td>{n} kg/ha</td></tr>
  <tr><td>P (Phosphorus)</td><td>{p} kg/ha</td></tr>
  <tr><td>K (Potassium)</td><td>{k} kg/ha</td></tr>
  <tr><td>Soil pH</td><td>{ph}</td></tr>
  <tr><td>🌡️ Temperature</td><td>{temp} °C</td></tr>
  <tr><td>🌧️ Rainfall</td><td>{rainfall} mm</td></tr>
  <tr><td>💧 Humidity</td><td>{humidity} %</td></tr>
</table>

<h2>💡 {T['tips_header']}</h2>
{"".join(f'<div class="tip">{t}</div>' for t in tips)}

<footer>
  🌾 Agri AI — Crop Yield Intelligence &nbsp;|&nbsp; Developed by <b>Ritvik</b>
</footer>
</body>
</html>"""

                        report_bytes = report_html.encode('utf-8')
                        st.download_button(
                            label=T.get("report_download", "📥 Download Report (HTML)"),
                            data=report_bytes,
                            file_name=f"AgriAI_Report_{crop}_{state}_{year}.html",
                            mime="text/html",
                            use_container_width=True,
                        )

                    except ValueError as e:
                        st.error(f"❌ {str(e)}")

            else:
                st.markdown("""
                <div style="text-align:center; padding:3rem 1rem; color:#888;">
                    <div style="font-size:4rem;">🌾</div>
                    <div style="font-size:1rem; margin-top:0.5rem;">Fill in the details and click Predict</div>
                </div>
                """, unsafe_allow_html=True)

                # Show model stats
                if bundle:
                    st.markdown(f'<div class="section-title">{T["model_stats"]}</div>', unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("R² Score", f"{bundle.get('r2_log', 0)*100:.1f}%")
                    m2.metric("MAE", f"{bundle.get('mae', 0):.1f} kg/ha")
                    m3, m4 = st.columns(2)
                    m3.metric("States", len(bundle.get('states', [])))
                    m4.metric("Crops", len(bundle.get('crops', [])))

# ══════════════════════════════════════════════
#  TAB 2 — CROP RECOMMENDATIONS
# ══════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-title">{T["recommend_header"]}</div>', unsafe_allow_html=True)

    all_states = list(REGIONAL_CROPS.keys())
    if bundle:
        all_states = sorted(set(all_states + bundle.get('states', [])))

    rc1, rc2, rc3 = st.columns([1.5, 1.5, 1])
    with rc1:
        rec_state = st.selectbox(T["recommend_state"], all_states, key="rec_state")
    with rc2:
        rec_season = st.selectbox(T["recommend_season"], ["Kharif", "Rabi", "Whole Year"], key="rec_season")
    with rc3:
        st.markdown("<br>", unsafe_allow_html=True)
        rec_clicked = st.button(T["recommend_btn"], key="rec_btn")

    if rec_clicked:
        crops_list = REGIONAL_CROPS.get(rec_state, {}).get(rec_season, None)
        if not crops_list:
            crops_list = DEFAULT_CROPS.get(rec_season, DEFAULT_CROPS["Kharif"])

        st.markdown(f"### {T['top_crops']} — {rec_state} ({rec_season})")
        st.markdown("---")

        cols = st.columns(min(len(crops_list), 3))
        for i, c in enumerate(crops_list):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="crop-card">
                    <div style="font-size:2rem">{c['icon']}</div>
                    <div class="crop-name">{c['crop']}</div>
                    <div class="crop-reason">{c['reason']}</div>
                    <div class="crop-yield">📊 {c['avg_yield']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Climate info
        st.markdown("---")
        climate_info = {
            "Punjab": "🌡️ Hot summers, cold winters | 🌧️ 400–700mm rainfall | 🪨 Alluvial & sandy loam",
            "Maharashtra": "🌡️ Tropical | 🌧️ 400–2000mm | 🪨 Black cotton (Vertisol)",
            "Tamil Nadu": "🌡️ Tropical hot | 🌧️ 900–2000mm | 🪨 Red loam & alluvial",
            "Uttar Pradesh": "🌡️ Semi-arid | 🌧️ 600–1000mm | 🪨 Alluvial (Indo-Gangetic)",
            "Rajasthan": "🌡️ Arid/Semi-arid | 🌧️ 100–500mm | 🪨 Sandy & loamy",
            "West Bengal": "🌡️ Tropical humid | 🌧️ 1200–4000mm | 🪨 Alluvial delta",
            "Kerala": "🌡️ Tropical | 🌧️ 2000–4000mm | 🪨 Laterite & alluvial",
            "Andhra Pradesh": "🌡️ Tropical | 🌧️ 800–1200mm | 🪨 Red loam & black",
        }
        info = climate_info.get(rec_state, "🌡️ Varied climate | 🌧️ Moderate rainfall | 🪨 Mixed soil types")
        st.markdown(f'<div class="info-box"><b>🗺️ {rec_state} — Climate & Soil:</b><br>{info}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#888;">
            <div style="font-size:4rem">🌿</div>
            <div>Select your state and season, then click the button</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 3 — BATCH PREDICT
# ══════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-title">{T["batch_header"]}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    📌 Upload a CSV file with columns: <b>state, crop, season, year, fertilizer, pesticide, n, p, k, ph, avg_temp_c, total_rainfall_mm, avg_humidity_percent</b>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(T["upload_csv"], type=["csv"])

    if uploaded and bundle:
        df_input = pd.read_csv(uploaded)
        st.write("**Preview:**", df_input.head())

        if st.button("▶️ Run Batch Prediction"):
            preds = []
            errors = []
            progress = st.progress(0)
            for i, row in df_input.iterrows():
                try:
                    y = predict(
                        bundle,
                        state=str(row.get('state', '')).strip().title(),
                        crop=str(row.get('crop', '')).strip().title(),
                        season=str(row.get('season', '')).strip(),
                        year=int(row.get('year', 2020)),
                        fertilizer=float(row.get('fertilizer', 100)),
                        pesticide=float(row.get('pesticide', 1)),
                        n=float(row.get('n', 60)),
                        p=float(row.get('p', 30)),
                        k=float(row.get('k', 30)),
                        ph=float(row.get('ph', 6.5)),
                        avg_temp_c=float(row.get('avg_temp_c', 25)),
                        total_rainfall_mm=float(row.get('total_rainfall_mm', 1000)),
                        avg_humidity_percent=float(row.get('avg_humidity_percent', 60)),
                    )
                    preds.append(y)
                    errors.append(None)
                except Exception as e:
                    preds.append(None)
                    errors.append(str(e))
                progress.progress((i + 1) / len(df_input))

            df_input['predicted_yield_kg_ha'] = preds
            df_input['error'] = errors
            st.success(f"✅ Done! {sum(p is not None for p in preds)}/{len(preds)} predictions successful.")
            st.dataframe(df_input)

            csv_out = df_input.to_csv(index=False).encode('utf-8')
            st.download_button(T["download_results"], csv_out, "batch_predictions.csv", "text/csv")

# ══════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════
with tab4:
    st.markdown(f"""
    <div class="app-header" style="margin-bottom:1.5rem;">
        <h1 style="font-size:1.6rem;color:#000000;font-weight:900;">ℹ️ {T.get('about_header', 'About This App')}</h1>
        <p style="color:#000000;font-weight:700;">Agri AI — AI-powered crop yield intelligence for Indian farmers</p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-icon">🤖</div>
            <div class="about-card-title">Machine Learning Model</div>
            <div class="about-card-body">
                <b>Algorithm:</b> Gradient Boosting Regressor<br>
                <b>Trees:</b> 600 &nbsp;|&nbsp; <b>LR:</b> 0.04<br>
                <b>R² Score:</b> 95.73% (log space)<br>
                <b>Data:</b> 1997–2020<br>
                <b>Coverage:</b> 29 Indian States
            </div>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-icon">📊</div>
            <div class="about-card-title">Features Used</div>
            <div class="about-card-body">
                🪨 <b>Soil:</b> N, P, K, pH<br>
                🌤️ <b>Weather:</b> Temp, Rainfall, Humidity<br>
                🧪 <b>Inputs:</b> Fertilizer, Pesticide<br>
                📅 <b>Temporal:</b> Year, Lag Yields<br>
                🔢 <b>Derived:</b> NPK total, Rain×Temp
            </div>
        </div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-icon">🌍</div>
            <div class="about-card-title">Languages Supported</div>
            <div class="about-card-body">
                हिंदी • বাংলা • தமிழ் • తెలుగు<br>
                ಕನ್ನಡ • മലയാളം • ਪੰਜਾਬੀ<br>
                मराठी • ગુજરાતી • ଓଡ଼ିଆ<br>
                اردو • English<br>
                <span style="color:#4CAF50;font-weight:700;">13 Languages Total</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-icon">📁</div>
            <div class="about-card-title">Required Files</div>
            <div class="about-card-body">
                📦 <code>crop_yield_model.pkl</code> — trained model<br>
                📊 <code>crop_yield.csv</code> — for retraining<br>
                🪨 <code>state_soil_data.csv</code><br>
                🌦️ <code>state_weather_data_1997_2020.csv</code><br>
                🌾 <code>crop_production.csv</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-icon">🗺️</div>
            <div class="about-card-title">States & Union Territories</div>
            <div class="about-card-body">
                Andhra Pradesh • Assam • Bihar • Chhattisgarh<br>
                Gujarat • Haryana • Himachal Pradesh<br>
                Jharkhand • Karnataka • Kerala • M.P.<br>
                Maharashtra • Manipur • Meghalaya<br>
                Nagaland • Odisha • Punjab • Rajasthan<br>
                Sikkim • Tamil Nadu • Telangana • Tripura<br>
                U.P. • Uttarakhand • West Bengal<br>
                <span style="color:#4CAF50;font-weight:700;">+ More States / UTs in Dataset</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if bundle:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{T["model_stats"]}</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² (Log)",    f"{bundle.get('r2_log',0)*100:.2f}%")
        c2.metric("MAE",         f"{bundle.get('mae',0):.2f} kg/ha")
        c3.metric("Train rows",  f"{bundle.get('train_size',0):,}")
        c4.metric("Test rows",   f"{bundle.get('test_size',0):,}")

        if 'feature_importance' in bundle:
            st.markdown("#### 📊 Top Feature Importances")
            fi = pd.Series(bundle['feature_importance']).sort_values(ascending=False).head(10)
            st.bar_chart(fi)

# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <span>🌾 Agri AI — Crop Yield Intelligence</span>
    &nbsp;|&nbsp;
    <span>Developed with ❤️ by <b>Ritvik</b></span>
    &nbsp;|&nbsp;
    <span>AI-powered agriculture for Indian farmers</span>
</div>
""", unsafe_allow_html=True)