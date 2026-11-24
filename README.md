# RetainCraft — Review Triage & Retention Studio

**RetainCraft** is an AI-powered customer review triage and retention web application built with **Python**, **Scikit-Learn**, and **Streamlit**. It helps customer support and retention teams prioritize negative reviews, estimate churn risk, calculate fair INR goodwill retention credits, and uncover actionable retention insights from single reviews or bulk CSV uploads.

## Key Features
- **Single-Review & Bulk CSV Triage:** Paste customer reviews line by line or upload a CSV file (`sample_unlabeled_1000.csv`) for instant batch scoring and downloadable priority queues.
- **Urgency & Churn Risk Scoring:** Predicts review urgency (`Positive / Low Urgency`, `Neutral / Medium Urgency`, `Negative / High Urgency`) and computes a weighted **Churn Risk (%)** score combining model probabilities with delivery delay.
- **Revenue Protection & INR Goodwill Estimator:** Tracks **At-Risk Revenue Value (₹)** and recommends bounded **INR goodwill credits** (`₹50` to `₹1,000`) to retain elevated-risk customers.
- **Actionable Retention Suggestions & Visuals:** Generates targeted next-step recommendations alongside interactive **Matplotlib** and **Seaborn** priority charts.

## Project Structure
- `app.py` — Main Streamlit web application, custom dark-mode UI, scoring engine, and interactive charts.
- `retaincraft_triage_model.joblib` — Trained Scikit-Learn text and tabular triage pipeline.
- `capstone2.ipynb` — End-to-end exploratory data analysis, preprocessing, model training, and evaluation notebook.
- `requirements.txt` — Python dependencies required to run and deploy the app.
- `sample_unlabeled_1000.csv` — Sample dataset for testing bulk CSV triage in the app.

## Tech Stack
- **Frontend:** Streamlit, Custom HTML/CSS & SVG
- **Machine Learning & Data:** Python, Scikit-Learn, Joblib, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Run Locally
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
    streamlit run app.py
 
