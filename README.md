# ChurnSense AI – Candidate Analytics


ChurnSense AI is an AI-powered Candidate Churn Prediction & Reason Analysis Dashboard. It leverages machine learning to proactively identify candidates at risk of dropping out (churning) during or before their training/induction sessions. 

The system analyzes candidate profiles, payment histories, and executive call logs to identify churn risks, highlight key drivers (like skipping induction or lack of call engagement), and provide actionable recommendations.

## Project Structure

The workspace is organized to keep data, generated assets, and application code separated:

```text
churn_prediction/
├── data/                               # Input data files
│   ├── Candidate Profile.csv           # Candidate demographics and course details
│   ├── Call log.csv                    # Communication history and call remarks
│   └── Executive Profile.csv           # Details of assigned executives
├── outputs/                            # Model artifacts and generated reports
│   ├── churn_prediction_model.pkl      # Trained ML model (e.g. XGBoost)
│   ├── candidates_with_suggested_reasons.csv # Inference outputs and insights
│   ├── churn_reasons.csv               # Aggregated churn statistics
│   └── *.png                           # Visualizations (ROC, Confusion Matrix, etc.)
├── app.py                              # Streamlit Dashboard application
├── model.py                            # ML Training pipeline and feature engineering
├── requirement.txt                     # Python dependencies
└── .gitignore                          # Git ignore rules (venv, etc.)
```

## Features

1. **Machine Learning Pipeline (`model.py`)**:
   - Advanced feature engineering: Extracts call durations, frequencies, and NLP keyword flags from call remarks (e.g., "interested", "unreachable").
   - Class Imbalance Handling: Automatically evaluates and applies techniques like SMOTE, class weighting, or over/under-sampling.
   - Model Evaluation: Compares Logistic Regression, Random Forest, XGBoost, and more, tuning the best model via GridSearchCV.
   
2. **Interactive Streamlit Dashboard (`app.py`)**:
   - **Overview**: High-level KPIs, churn rates, and data source summaries.
   - **Candidate Explorer**: Deep-dive into individual candidate risk levels and specific reasons for potential churn.
   - **Call Log & Payment Analysis**: Visual insights into communication patterns and fee payment ratios.
   - **Live Predictor**: A dynamic form to input a specific scenario and receive a real-time churn probability and suggested action.
   - **Model Performance**: Transparency into the algorithm's accuracy, feature importance, and historical confusion matrices.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eldhosekroy/churn_prediction.git
   cd churn_prediction
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```powershell
   # Windows
   py -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

4. **Run the Dashboard:**
   ```bash
   python -m streamlit run app.py
   ```
   *The dashboard will automatically open in your default web browser.*

## Retraining the Model

If you receive new data (updated CSVs in the `data/` folder), you can retrain the model and regenerate all insights by simply running:

```bash
python model.py
```

This will automatically overwrite the `.pkl` model and visual assets inside the `outputs/` folder. Reload the Streamlit app to reflect the new predictions.