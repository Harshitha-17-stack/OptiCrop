"""
app.py
------
Flask web server for OptiCrop — the Smart Agricultural Production
Optimization Engine. Loads the trained model bundle at startup and
serves three pages: home, about, and the crop-recommendation form.
"""

import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from crop_info import get_crop_info

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load model assets once at startup
# ---------------------------------------------------------------------
MODEL_PATH = "model/model.pkl"
METRICS_PATH = "model/metrics.json"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
encoder = bundle["encoder"]
feature_columns = bundle["feature_columns"]
champion_name = bundle.get("champion_name", "Random Forest")
all_models = bundle.get("all_models", {})

try:
    with open(METRICS_PATH) as f:
        MODEL_METRICS = json.load(f)
except FileNotFoundError:
    MODEL_METRICS = bundle.get("metrics", {})

FIELD_SPECS = [
    {"key": "nitrogen",    "label": "Nitrogen (N)",    "unit": "kg/ha", "min": 0,  "max": 140, "step": 1,   "placeholder": "90"},
    {"key": "phosphorous", "label": "Phosphorous (P)", "unit": "kg/ha", "min": 0,  "max": 145, "step": 1,   "placeholder": "42"},
    {"key": "potassium",   "label": "Potassium (K)",   "unit": "kg/ha", "min": 0,  "max": 205, "step": 1,   "placeholder": "43"},
    {"key": "temperature", "label": "Temperature",     "unit": "°C",    "min": 0,  "max": 50,  "step": 0.1, "placeholder": "20.9"},
    {"key": "humidity",    "label": "Humidity",        "unit": "%",     "min": 0,  "max": 100, "step": 0.1, "placeholder": "82.0"},
    {"key": "ph",          "label": "Soil pH",         "unit": "pH",    "min": 0,  "max": 14,  "step": 0.1, "placeholder": "6.5"},
    {"key": "rainfall",    "label": "Rainfall",        "unit": "mm",    "min": 0,  "max": 300, "step": 0.1, "placeholder": "202.9"},
]


@app.route("/")
def home():
    return render_template("home.html", active_page="home")


@app.route("/about")
def about():
    return render_template(
        "about.html",
        active_page="about",
        metrics=MODEL_METRICS,
        champion_name=champion_name,
    )


@app.route("/findyourcrop", methods=["GET", "POST"])
def findyourcrop():
    result = None
    form_values = {f["key"]: f["placeholder"] for f in FIELD_SPECS}

    if request.method == "POST":
        try:
            form_values = {f["key"]: request.form.get(f["key"], "") for f in FIELD_SPECS}

            N = float(request.form["nitrogen"])
            P = float(request.form["phosphorous"])
            K = float(request.form["potassium"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            features = pd.DataFrame(
                [[N, P, K, temperature, humidity, ph, rainfall]],
                columns=feature_columns,
            )
            features_scaled = scaler.transform(features)

            prediction_idx = model.predict(features_scaled)[0]
            recommended_crop = encoder.inverse_transform([prediction_idx])[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_scaled)[0]
                confidence = round(float(np.max(proba)) * 100, 1)

            info = get_crop_info(recommended_crop)

            # Check for low confidence alternative
            alternative = None
            if confidence is not None and confidence < 70.0:
                best_alt_crop = None
                best_alt_conf = 0.0
                best_alt_model = None

                for m_name, m in all_models.items():
                    if m_name == champion_name:
                        continue
                    if hasattr(m, "predict_proba"):
                        proba_m = m.predict_proba(features_scaled)[0]
                        idx_m = np.argmax(proba_m)
                        conf_m = round(float(np.max(proba_m)) * 100, 1)
                        crop_m = encoder.inverse_transform([idx_m])[0]

                        if crop_m != recommended_crop and conf_m >= 85.0:
                            if conf_m > best_alt_conf:
                                best_alt_crop = crop_m
                                best_alt_conf = conf_m
                                best_alt_model = m_name

                if best_alt_crop:
                    alt_info = get_crop_info(best_alt_crop)
                    alternative = {
                        "crop": best_alt_crop.capitalize(),
                        "emoji": alt_info["emoji"],
                        "category": alt_info["category"],
                        "note": alt_info["note"],
                        "confidence": best_alt_conf,
                        "model_name": best_alt_model,
                        "type": "same_conditions",
                        "info": alt_info
                    }
                else:
                    best_adj_crop = None
                    best_adj_conf = 0.0
                    best_adj_specs = None

                    from crop_info import CROP_INFO
                    for crop_name, details in CROP_INFO.items():
                        if crop_name == recommended_crop:
                            continue
                        opt = details.get("optimal_ranges", {})
                        if not opt:
                            continue

                        adj_N = opt["N"]["mean"]
                        adj_P = opt["P"]["mean"]
                        adj_K = opt["K"]["mean"]
                        adj_ph = opt["ph"]["mean"]

                        adj_features = pd.DataFrame(
                            [[adj_N, adj_P, adj_K, temperature, humidity, adj_ph, rainfall]],
                            columns=feature_columns
                        )
                        adj_features_scaled = scaler.transform(adj_features)

                        adj_pred_idx = model.predict(adj_features_scaled)[0]
                        adj_pred_crop = encoder.inverse_transform([adj_pred_idx])[0]

                        if adj_pred_crop == crop_name:
                            adj_proba = model.predict_proba(adj_features_scaled)[0]
                            adj_conf = round(float(np.max(adj_proba)) * 100, 1)
                            if adj_conf >= 85.0:
                                if adj_conf > best_adj_conf:
                                    best_adj_crop = crop_name
                                    best_adj_conf = adj_conf
                                    best_adj_specs = {
                                        "N": adj_N, "P": adj_P, "K": adj_K, "ph": adj_ph
                                    }

                    if best_adj_crop:
                        adj_info = get_crop_info(best_adj_crop)
                        alternative = {
                            "crop": best_adj_crop.capitalize(),
                            "emoji": adj_info["emoji"],
                            "category": adj_info["category"],
                            "note": adj_info["note"],
                            "confidence": best_adj_conf,
                            "type": "adjusted_soil",
                            "specs": best_adj_specs,
                            "info": adj_info
                        }

            # Generate range comparisons
            ranges = []
            param_mapping = [
                {"key": "nitrogen", "stat_key": "N", "label": "Nitrogen", "unit": "kg/ha", "max": 140.0, "user_val": N},
                {"key": "phosphorous", "stat_key": "P", "label": "Phosphorous", "unit": "kg/ha", "max": 145.0, "user_val": P},
                {"key": "potassium", "stat_key": "K", "label": "Potassium", "unit": "kg/ha", "max": 205.0, "user_val": K},
                {"key": "temperature", "stat_key": "temp", "label": "Temperature", "unit": "°C", "max": 50.0, "user_val": temperature},
                {"key": "humidity", "stat_key": "humidity", "label": "Humidity", "unit": "%", "max": 100.0, "user_val": humidity},
                {"key": "ph", "stat_key": "ph", "label": "Soil pH", "unit": "pH", "max": 14.0, "user_val": ph},
                {"key": "rainfall", "stat_key": "rainfall", "label": "Rainfall", "unit": "mm", "max": 300.0, "user_val": rainfall},
            ]

            for pm in param_mapping:
                opt_spec = info["optimal_ranges"].get(pm["stat_key"], {"min": 0, "max": pm["max"], "mean": pm["max"]/2})
                opt_min = opt_spec["min"]
                opt_max = opt_spec["max"]
                user_val = pm["user_val"]
                abs_max = pm["max"]

                # Calculate status
                if user_val < opt_min:
                    status = "low"
                elif user_val > opt_max:
                    status = "high"
                else:
                    status = "optimal"

                # Calculate percentages for slider
                opt_left = (opt_min / abs_max) * 100.0
                opt_width = ((opt_max - opt_min) / abs_max) * 100.0
                user_pct = (user_val / abs_max) * 100.0
                user_pct = max(0.0, min(100.0, user_pct)) # clamp

                ranges.append({
                    "label": pm["label"],
                    "unit": pm["unit"],
                    "user_val": user_val,
                    "opt_min": opt_min,
                    "opt_max": opt_max,
                    "status": status,
                    "opt_left_pct": round(opt_left, 1),
                    "opt_width_pct": round(opt_width, 1),
                    "user_pct": round(user_pct, 1)
                })

            result = {
                "crop": recommended_crop.capitalize(),
                "emoji": info["emoji"],
                "category": info["category"],
                "note": info["note"],
                "confidence": confidence if confidence is not None else 100.0,
                "info": info,
                "alternative": alternative,
                "ranges": ranges,
                "inputs": {
                    "N": N, "P": P, "K": K,
                    "temperature": temperature, "humidity": humidity,
                    "ph": ph, "rainfall": rainfall,
                },
            }
        except (KeyError, ValueError):
            result = {"error": "Please enter valid numeric readings for every field."}

    return render_template(
        "findyourcrop.html",
        active_page="findyourcrop",
        fields=FIELD_SPECS,
        values=form_values,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
