# app.py
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# === Muat model Logistic Regression ===
with open('logistic_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# === Muat model Random Forest ===
with open('model_random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Muat daftar fitur (sekarang SAMA untuk kedua model)
with open('lr_feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Gunakan daftar fitur yang sama untuk kedua model
base_features = feature_columns

@app.route('/')
def home():
    return render_template('index.html')

# Halaman Logistic Regression
@app.route('/logistic')
def logistic_page():
    return render_template('logistic.html', features=base_features)

# Halaman Random Forest
@app.route('/randomforest')
def rf_page():
    return render_template('random_forest.html', features=base_features)

# Prediksi untuk Logistic Regression (versi sederhana)
@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    try:
        # Ambil input (12 fitur dasar)
        data = {feat: float(request.form[feat]) for feat in base_features}
        df = pd.DataFrame([data])

        # TIDAK ADA one-hot, TIDAK ADA fitur interaksi — langsung prediksi
        df = df[base_features]

        proba = lr_model.predict_proba(df)[0][1]
        threshold = 0.60
        prediction = "Positif" if proba >= threshold else "Netral"
        target = 1 if prediction == "Positif" else 0

        return render_template(
            'logistic.html',
            features=base_features,
            prediction=prediction,
            target=target,
            confidence=f"{proba:.2%}",
            input_data=data
        )
    except Exception as e:
        return render_template('logistic.html', features=base_features, error=f"Error: {str(e)}")

# Prediksi untuk Random Forest
@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        # Ambil input (12 fitur dasar)
        data = {feat: float(request.form[feat]) for feat in base_features}
        df = pd.DataFrame([data])

        # Langsung prediksi
        df = df[base_features]

        proba = rf_model.predict_proba(df)[0][1]
        threshold = 0.60
        prediction = "Positif" if proba >= threshold else "Netral"
        target = 1 if prediction == "Positif" else 0

        return render_template(
            'random_forest.html',
            features=base_features,
            prediction=prediction,
            target=target,
            confidence=f"{proba:.2%}",
            input_data=data
        )
    except Exception as e:
        return render_template('random_forest.html', features=base_features, error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)