# OptiCrop — Smart Agricultural Production Optimization Engine

OptiCrop is a Machine Learning-based web application that recommends the most
suitable crop for a given soil and climate reading: Nitrogen (N), Phosphorous (P),
Potassium (K), temperature, humidity, pH, and rainfall.

It's built for three audiences:

- **Farmers** — get an instant crop recommendation for a soil/climate reading.
- **Researchers** — explore crop-environment relationships and model behaviour.
- **Policymakers** — use data-driven patterns to inform agricultural strategy.

## Project structure

```
OptiCrop/
├── dataset/
│   └── Crop_recommendation.csv     # 2,200 rows, 22 crop classes
├── model/
│   ├── model.pkl                   # champion model + scaler + label encoder
│   └── metrics.json                # comparison metrics for all trained models
├── templates/
│   ├── _nav.html                   # shared navigation partial
│   ├── _footer.html                # shared footer partial
│   ├── home.html                   # landing page
│   ├── about.html                  # project info + architecture + model metrics
│   └── findyourcrop.html           # soil-reading form + live result
├── static/
│   ├── css/style.css               # design system & page styles
│   ├── js/soilprofile.js           # live N-P-K visualization on the form
│   └── images/                     # logo & background SVGs
├── app.py                          # Flask application (routes + inference)
├── model_training.py               # trains & compares models, saves the champion
├── preprocessing.py                # data loading, cleaning, encoding, scaling
├── crop_info.py                    # per-crop emoji / category / agronomic note
└── requirements.txt
```

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate.bat
   # macOS/Linux: source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Already done) Train the model** — a trained `model/model.pkl` is included,
   but you can retrain it any time:

   ```bash
   python model_training.py
   ```

   This loads `dataset/Crop_recommendation.csv`, cleans and preprocesses it
   (`preprocessing.py`), trains Logistic Regression, Decision Tree, Random
   Forest, Naive Bayes, and SVM classifiers, compares them, and saves the
   best-performing model (features + scaler + label encoder bundled together)
   to `model/model.pkl`, plus a `model/metrics.json` comparison table.

4. **Run the app**

   ```bash
   python app.py
   ```

   Then open **http://127.0.0.1:5000**.

## Pages

- `/` — Home: project overview and the three usage scenarios.
- `/about` — About: architecture, tech stack, and model comparison table.
- `/findyourcrop` — Enter a soil/climate reading and get a crop recommendation,
  a confidence gauge, and a short agronomic note.

## Retraining on your own data

Replace `dataset/Crop_recommendation.csv` with your own file (same 8 columns:
`N, P, K, temperature, humidity, ph, rainfall, label`) and re-run
`python model_training.py`. The Flask app automatically picks up the new
`model/model.pkl` on the next restart.

## Deployment

A minimal `render.yaml` for platforms like Render:

```yaml
services:
  - type: web
    name: opticrop
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:app"
```
