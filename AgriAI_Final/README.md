# 🌾 Agri AI — Crop Yield Predictor

> AI-powered agricultural intelligence for Indian farmers  
> **Developed by Ritvik**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![Accuracy](https://img.shields.io/badge/Accuracy-95.7%25-brightgreen)
![Languages](https://img.shields.io/badge/Languages-22-green)
![States](https://img.shields.io/badge/States-29-orange)

---

## ✨ Features

- 🤖 **ML Model** — Gradient Boosting, 95.7% R² accuracy
- 🌐 **22 Languages** — All Indian 8th Schedule languages
- 🗺️ **29 States** — Complete Indian coverage
- 📊 **Interactive Charts** — Soil, Yield trend, Weather
- 📥 **Report Download** — HTML report in selected language
- 📋 **Batch Predict** — Upload CSV for bulk predictions
- 💡 **Crop Recommendations** — Region & season wise

### 🌐 Languages Supported
Hindi • Bengali • Tamil • Telugu • Kannada • Malayalam • Punjabi • Marathi • Gujarati • Odia • Urdu • Assamese • Maithili • Kashmiri • Manipuri • Nepali • Sindhi • Konkani • Dogri • Bodo • Santali • English

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/AgriAI.git
cd AgriAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run app (model already included!)
streamlit run app.py
```

> ✅ **`crop_yield_model.pkl` is already included** — no training needed!  
> Just install requirements and run the app directly.

---

## 🔄 Retrain Model (Optional)

If you want to retrain with real data:

```bash
# Download dataset from Kaggle:
# https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
# Save as crop_production.csv, then:

python train_model.py
streamlit run app.py
```

---

## 📁 Project Structure

```
AgriAI/
├── app.py                  ← Streamlit app (22 languages, charts, UI)
├── train_model.py          ← ML training script
├── requirements.txt        ← Dependencies
├── README.md               ← This file
└── crop_yield_model.pkl    ← Pre-trained model (ready to use!)
```

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Trees | 600 |
| R² Score | 95.7% |
| Data Period | 1997–2020 |
| States | 29 Indian States |
| Crops | 30+ varieties |

---

## ⚙️ Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

No matplotlib, no plotly — zero extra chart libraries!

---

## 🌐 Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → Select repo
4. Main file: `app.py`
5. Click **Deploy** 🚀

---

## 👨‍💻 Developer

**Ritvik**  
Built with ❤️ for Indian farmers 🌾
