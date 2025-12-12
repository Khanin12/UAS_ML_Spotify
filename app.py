# app.py
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# === Muat artefak dari training ===
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Fitur input dasar dari form HTML (sesuai dataset asli)
base_features = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'key', 'liveness', 'loudness',
    'mode', 'speechiness', 'tempo', 'time_signature'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/randomforest')
def rf_page():
    return render_template('rf.html', features=base_features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # === 1. Ambil input dari form sebagai float ===
        data = {}
        for feat in base_features:
            val = request.form[feat]
            data[feat] = float(val)  # Pastikan float (konsisten dengan training)

        df = pd.DataFrame([data])

        # === 2. Rekayasa fitur identik dengan training ===
        df = pd.get_dummies(
            df,
            columns=['key', 'time_signature'],
            prefix=['key', 'ts'],
            drop_first=True
        )

        # Tambahkan fitur interaksi
        df['mode_x_energy'] = df['mode'] * df['energy']
        df['mode_x_danceability'] = df['mode'] * df['danceability']
        df['energy_x_acousticness'] = df['energy'] * df['acousticness']

        # === 3. Alignment fitur ===
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]  # Urutkan sesuai training

        # === 4. Prediksi dengan threshold 60% ===
        proba = model.predict_proba(df)[0][1]
        threshold = 0.60

        if proba >= threshold:
            prediction_label = "Positif"
            target = 1
        else:
            prediction_label = "Netral"   # ← Ganti dari "Negatif" ke "Netral"
            target = 0

        # === 5. Kirim semua ke template ===
        return render_template(
            'rf.html',
            features=base_features,
            prediction=prediction_label,
            target=target,                # ← Tambahkan target (0/1)
            confidence=f"{proba:.2%}",
            input_data=data
        )

    except Exception as e:
        return render_template(
            'rf.html',
            features=base_features,
            error=f"Error: {str(e)}"
        )

if __name__ == '__main__':
    app.run(debug=True)