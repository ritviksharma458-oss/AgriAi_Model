"""
=============================================================
  AGRI AI — CROP YIELD MODEL TRAINING SCRIPT
  Generates: crop_yield_model.pkl
  Author: Ritvik

  HOW TO USE:
  -----------
  OPTION A (Real Data — Better Accuracy):
    1. Download from Kaggle:
       https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
    2. Save as crop_production.csv in this folder
    3. Run: python train_model.py

  OPTION B (No dataset needed — works directly):
    Just run: python train_model.py
    Synthetic data will be generated automatically.

  OUTPUT:
    crop_yield_model.pkl  ← loaded by app.py

  THEN RUN APP:
    streamlit run app.py
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 60)
print("  AGRI AI — Crop Yield Model Training")
print("  Developed by Ritvik")
print("=" * 60)

# ─────────────────────────────────────────────
#  FEATURES — must match app.py exactly
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
#  STATE CLIMATE + SOIL PROFILES
# ─────────────────────────────────────────────
STATE_DEFAULTS = {
    'Andhra Pradesh':    dict(temp=30.0, rain=1000, hum=72, n=75,  p=35, k=35, ph=6.5),
    'Assam':             dict(temp=26.0, rain=2000, hum=82, n=60,  p=28, k=30, ph=5.8),
    'Bihar':             dict(temp=27.0, rain=1100, hum=74, n=70,  p=32, k=32, ph=7.0),
    'Chhattisgarh':      dict(temp=28.0, rain=1300, hum=72, n=65,  p=30, k=30, ph=6.5),
    'Gujarat':           dict(temp=30.0, rain=700,  hum=65, n=65,  p=30, k=30, ph=7.2),
    'Haryana':           dict(temp=26.0, rain=550,  hum=60, n=85,  p=40, k=40, ph=7.5),
    'Himachal Pradesh':  dict(temp=18.0, rain=1100, hum=68, n=60,  p=28, k=28, ph=6.8),
    'Jharkhand':         dict(temp=27.0, rain=1200, hum=72, n=58,  p=27, k=27, ph=6.2),
    'Karnataka':         dict(temp=28.0, rain=1100, hum=72, n=70,  p=33, k=33, ph=6.5),
    'Kerala':            dict(temp=29.0, rain=3000, hum=85, n=65,  p=30, k=40, ph=5.5),
    'Madhya Pradesh':    dict(temp=28.0, rain=1000, hum=68, n=68,  p=32, k=32, ph=7.0),
    'Maharashtra':       dict(temp=28.0, rain=900,  hum=70, n=70,  p=33, k=33, ph=7.2),
    'Manipur':           dict(temp=24.0, rain=1500, hum=78, n=58,  p=26, k=26, ph=6.0),
    'Meghalaya':         dict(temp=22.0, rain=2500, hum=82, n=55,  p=25, k=25, ph=5.5),
    'Nagaland':          dict(temp=23.0, rain=1800, hum=80, n=55,  p=25, k=25, ph=5.8),
    'Odisha':            dict(temp=29.0, rain=1500, hum=78, n=68,  p=32, k=32, ph=6.5),
    'Punjab':            dict(temp=25.0, rain=600,  hum=65, n=90,  p=45, k=45, ph=7.8),
    'Rajasthan':         dict(temp=33.0, rain=300,  hum=45, n=55,  p=25, k=25, ph=8.0),
    'Sikkim':            dict(temp=20.0, rain=2000, hum=80, n=55,  p=25, k=25, ph=5.5),
    'Tamil Nadu':        dict(temp=30.0, rain=1100, hum=75, n=72,  p=34, k=34, ph=6.5),
    'Telangana':         dict(temp=30.0, rain=950,  hum=70, n=70,  p=33, k=33, ph=6.8),
    'Tripura':           dict(temp=27.0, rain=2000, hum=82, n=60,  p=28, k=28, ph=5.8),
    'Uttar Pradesh':     dict(temp=27.0, rain=850,  hum=68, n=80,  p=38, k=38, ph=7.5),
    'Uttarakhand':       dict(temp=22.0, rain=1200, hum=70, n=65,  p=30, k=30, ph=6.8),
    'West Bengal':       dict(temp=28.0, rain=1600, hum=80, n=72,  p=34, k=34, ph=6.2),
    'Jammu And Kashmir': dict(temp=15.0, rain=800,  hum=62, n=60,  p=28, k=28, ph=7.0),
    'Goa':               dict(temp=28.0, rain=2900, hum=83, n=60,  p=28, k=35, ph=5.8),
    'Arunachal Pradesh': dict(temp=21.0, rain=2500, hum=82, n=55,  p=25, k=25, ph=5.5),
    'Mizoram':           dict(temp=24.0, rain=2000, hum=80, n=55,  p=25, k=25, ph=5.5),
}

# Realistic crop yield ranges (kg/ha)
CROP_YIELDS = {
    'Rice':        (2000, 5000),   'Wheat':       (2500, 5500),
    'Maize':       (1800, 4000),   'Soyabean':    (800,  1800),
    'Sugarcane':   (50000,100000), 'Cotton':      (400,  900),
    'Groundnut':   (1000, 2500),   'Potato':      (15000,35000),
    'Mustard':     (800,  1800),   'Jowar':       (600,  1500),
    'Bajra':       (700,  1600),   'Gram':        (700,  1600),
    'Sunflower':   (800,  1800),   'Tobacco':     (1500, 2800),
    'Arecanut':    (2000, 4000),   'Jute':        (2000, 3500),
    'Coconut':     (10000,20000),  'Onion':       (15000,30000),
    'Tomato':      (20000,40000),  'Banana':      (20000,40000),
    'Mango':       (5000, 12000),  'Turmeric':    (4000, 8000),
    'Pepper':      (1000, 2500),   'Cardamom':    (500,  1200),
    'Coffee':      (500,  1500),   'Tea':         (1500, 3000),
    'Rubber':      (1000, 2500),   'Cashew':      (800,  1800),
    'Ragi':        (1000, 2500),   'Barley':      (1500, 3500),
}

SEASONS = ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter', 'Autumn']


# ─────────────────────────────────────────────
#  GENERATE SYNTHETIC DATA
# ─────────────────────────────────────────────
def generate_synthetic(n_rows=12000):
    print(f"\n📊 Generating {n_rows:,} synthetic training records...")
    states = list(STATE_DEFAULTS.keys())
    crops  = list(CROP_YIELDS.keys())
    rows   = []

    for _ in range(n_rows):
        state  = np.random.choice(states)
        crop   = np.random.choice(crops)
        season = np.random.choice(SEASONS)
        year   = int(np.random.randint(1997, 2021))
        clim   = STATE_DEFAULTS[state]
        ymin, ymax = CROP_YIELDS.get(crop, (1000, 4000))

        temp  = clim['temp']  + np.random.normal(0, 2.5)
        rain  = max(50, clim['rain']  + np.random.normal(0, clim['rain'] * 0.18))
        hum   = np.clip(clim['hum']   + np.random.normal(0, 5), 20, 100)
        n_val = max(10, clim['n']     + np.random.normal(0, 12))
        p_val = max(5,  clim['p']     + np.random.normal(0, 8))
        k_val = max(5,  clim['k']     + np.random.normal(0, 8))
        ph    = np.clip(clim['ph']    + np.random.normal(0, 0.4), 3.5, 9.5)
        fert  = np.random.uniform(50, 300)
        pest  = np.random.uniform(0.5, 15)

        rain_f  = min(rain / 1000, 2.0) ** 0.5
        temp_f  = max(0.3, 1.0 - abs(temp - 27) / 30)
        npk_f   = (n_val + p_val + k_val) / 200
        fert_f  = np.log1p(fert) / np.log1p(200)
        base    = (ymin + ymax) / 2
        yval    = base * rain_f * temp_f * npk_f * fert_f
        yval   *= np.random.uniform(0.75, 1.25)
        yval    = max(50, yval)

        lag1      = yval * np.random.uniform(0.85, 1.15)
        lag2      = yval * np.random.uniform(0.80, 1.20)
        rain_lag1 = rain + np.random.normal(0, 80)

        rows.append(dict(
            state=state, crop=crop, season=season, year=year,
            fertilizer=fert, pesticide=pest,
            n=n_val, p=p_val, k=k_val, ph=ph,
            avg_temp_c=temp, total_rainfall_mm=rain,
            avg_humidity_percent=hum,
            yield_kg_per_ha=yval,
            yield_lag1=lag1, yield_lag2=lag2, rain_lag1=rain_lag1,
        ))

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    for fname in ['crop_production.csv', 'crop_yield.csv', 'crop_data.csv']:
        if os.path.exists(fname):
            print(f"\n✅ Found {fname} — using REAL dataset!")
            df = pd.read_csv(fname)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            return df, True
    print("\n⚠️  No CSV found. Using synthetic demo data.")
    print("   For real training, download from:")
    print("   https://www.kaggle.com/datasets/abhinand05/crop-production-in-india\n")
    return generate_synthetic(), False


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    rename = {
        'state_name': 'state', 'crop_year': 'year',
        'area': 'area_ha', 'production': 'production_tonnes',
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    if 'yield_kg_per_ha' not in df.columns:
        if 'production_tonnes' in df.columns and 'area_ha' in df.columns:
            df['yield_kg_per_ha'] = (df['production_tonnes'] * 1000) / (df['area_ha'].replace(0, np.nan) + 1)
        else:
            raise ValueError("Dataset needs 'yield_kg_per_ha' OR ('production_tonnes' + 'area_ha')")

    defaults = dict(
        fertilizer=120, pesticide=2.5,
        n=70, p=33, k=33, ph=6.8,
        avg_temp_c=27, total_rainfall_mm=900, avg_humidity_percent=68,
        yield_lag1=0, yield_lag2=0, rain_lag1=900,
    )
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    for col in ['state', 'crop', 'season']:
        df[col] = df[col].astype(str).str.strip().str.title()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2010).astype(int)

    df = df[df['yield_kg_per_ha'] > 10]
    df = df[df['yield_kg_per_ha'] < df['yield_kg_per_ha'].quantile(0.99)]
    df.dropna(subset=['yield_kg_per_ha', 'state', 'crop', 'season'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    le_state  = LabelEncoder().fit(df['state'])
    le_crop   = LabelEncoder().fit(df['crop'])
    le_season = LabelEncoder().fit(df['season'])
    df['state_enc']  = le_state.transform(df['state'])
    df['crop_enc']   = le_crop.transform(df['crop'])
    df['season_enc'] = le_season.transform(df['season'])

    df['log_fertilizer']     = np.log1p(df['fertilizer'])
    df['log_pesticide']      = np.log1p(df['pesticide'])
    df['npk_total']          = df['n'] + df['p'] + df['k']
    df['npk_ratio_np']       = df['n'] / (df['p'] + 1)
    df['rain_temp_interact'] = df['total_rainfall_mm'] * df['avg_temp_c']
    df['fert_per_rain']      = df['fertilizer'] / (df['total_rainfall_mm'] + 1)
    df['yield_lag1_log']     = np.log1p(df['yield_lag1'])
    df['yield_lag2_log']     = np.log1p(df['yield_lag2'])
    df['yield_ma2_log']      = (df['yield_lag1_log'] + df['yield_lag2_log']) / 2
    df['yield_diff']         = df['yield_lag1'] - df['yield_lag2']
    df['rainfall_diff']      = df['total_rainfall_mm'] - df['rain_lag1']
    df['temp_diff']          = 0.0
    df['log_yield']          = np.log1p(df['yield_kg_per_ha'])

    return df, le_state, le_crop, le_season


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def train():
    df_raw, is_real = load_data()
    print(f"   Rows loaded: {len(df_raw):,}")

    print("\n⚙️  Engineering features...")
    df, le_state, le_crop, le_season = engineer_features(df_raw)
    print(f"   States  : {len(le_state.classes_)}")
    print(f"   Crops   : {len(le_crop.classes_)}")
    print(f"   Seasons : {len(le_season.classes_)}")
    print(f"   Clean rows: {len(df):,}")

    X = df[FEATURES]
    y = df['log_yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n🤖 Training Gradient Boosting (600 trees)...")
    print("   Please wait 1-3 minutes...")
    model = GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.04,
        max_depth=5, min_samples_split=10,
        min_samples_leaf=5, subsample=0.85,
        max_features='sqrt', random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

    print(f"\n📈 Results:")
    print(f"   R² Score  : {r2*100:.2f}%")
    print(f"   MAE       : {mae:.1f} kg/ha")

    fi = dict(zip(FEATURES, model.feature_importances_))
    top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]
    print(f"\n🔍 Top 5 Features:")
    for feat, imp in top5:
        print(f"   {feat:<25} {imp:.4f}")

    bundle = {
        'model':              model,
        'le_state':           le_state,
        'le_crop':            le_crop,
        'le_season':          le_season,
        'states':             sorted(le_state.classes_.tolist()),
        'crops':              sorted(le_crop.classes_.tolist()),
        'seasons':            sorted(le_season.classes_.tolist()),
        'features':           FEATURES,
        'r2_log':             r2,
        'mae':                mae,
        'train_size':         len(X_train),
        'test_size':          len(X_test),
        'feature_importance': fi,
        'is_real_data':       is_real,
    }

    out = 'crop_yield_model.pkl'
    joblib.dump(bundle, out, compress=3)
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"\n✅ Saved: {out}  ({mb:.1f} MB)")
    print(f"\n🚀 Now run:")
    print(f"   streamlit run app.py")
    print("=" * 60)


if __name__ == '__main__':
    train()
