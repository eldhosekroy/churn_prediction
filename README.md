# ChurnSense AI – Candidate Analytics & Call Analyzer Pro

ChurnSense AI is an AI-powered Candidate Churn Prediction & Reason Analysis platform. It leverages machine learning to proactively identify candidates at risk of dropping out (churning) during or before their training/induction sessions. 

Recently, the system has been upgraded to include an advanced **Call Analyzer Pro** tool for transcribing, translating (Malayalam to English), and generating AI-driven insights from call recordings.

## Project Structure

The workspace is organized to keep data, generated assets, and application code separated:

```text
churn_prediction/
├── data/                               # Input data files
│   ├── Candidate Profile.csv           # Candidate demographics and course details
│   ├── Call log.csv                    # Communication history and call remarks
│   └── Executive Profile.csv           # Details of assigned executives
├── outputs/                            # Model artifacts and generated reports
│   ├── prediction_model.pkl            # Trained ML model (e.g. XGBoost)
│   ├── candidates_with_suggested_reasons.csv # Inference outputs and insights
│   ├── churn_reasons.csv               # Aggregated churn statistics
│   └── *.png                           # Visualizations (ROC, Confusion Matrix, etc.)
├── call_analyzer/                      # AI-Powered Call Analysis & Transcription Submodule
│   ├── app.py                          # Dedicated Streamlit app for call analysis
│   └── readme.md                       # Documentation for the Call Analyzer
├── app.py                              # Legacy Streamlit Dashboard application
├── dashboard.py                        # NEW AI-Powered Streamlit Dashboard (Supabase integrated)
├── model.py                            # ML Training pipeline and feature engineering
├── database_pipeline.py                # Supabase integration and data logging routines
├── llm_integration.py                  # Generative AI (Gemini/Groq) integrations for insights
├── prediction_models.py                # Machine learning prediction wrappers
├── requirements.txt                    # Pinned Python dependencies (scikit-learn 1.8.0, xgboost 3.2.0)
└── .env.example                        # Example environment variables required for deployment
```

## Core Features

1. **AI-Powered Streamlit Dashboard (`dashboard.py`)**:
   - **Overview**: High-level KPIs, churn rates, and data source summaries.
   - **Candidate Explorer**: Deep-dive into individual candidate risk levels and specific reasons for potential churn.
   - **Live Predictor**: A dynamic form to input a specific scenario and receive a real-time churn probability and suggested action.
   - **Supabase Integration**: Automatically logs queries, candidates, and call insights into a Supabase PostgreSQL database.

2. **Call Analyzer Pro (`call_analyzer/`)**:
   - Upload call recordings (MP3, WAV, etc.) or paste transcriptions directly.
   - Automatic language detection and translation (Malayalam to English).
   - Generates concise call remarks and extracts keywords, entities, and sentiments.

3. **Machine Learning Pipeline (`model.py` & `prediction_models.py`)**:
   - Advanced feature engineering: Extracts call durations, frequencies, and NLP keyword flags.
   - Handles class imbalance using modern techniques.
   - Model Evaluation: Compares Logistic Regression, Random Forest, XGBoost, and more, tuning the best model via GridSearchCV.

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
   pip install -r requirements.txt
   ```

4. **Environment Variables (.env)**
   Copy the `.env.example` file to a new file named `.env` and fill in your API keys (Gemini, Groq, HuggingFace, Supabase):
   ```bash
   cp .env.example .env
   ```

5. **Run the Dashboard:**
   ```bash
   streamlit run dashboard.py
   ```
   *The dashboard will automatically open in your default web browser.*

## Deployment to Streamlit Cloud

1. Connect your GitHub repository to [Streamlit Community Cloud](https://streamlit.io/cloud).
2. Set the main file path to `dashboard.py`.
3. In the Streamlit App Settings -> **Secrets**, paste the contents of your `.env` file formatted as TOML. For example:
   ```toml
   GEMINI_MODEL = "gemini-2.5-flash"
   SUPABASE_URL = "https://your-project-id.supabase.co"
   SUPABASE_KEY = "your-supabase-anon-key"
   # ... add your other API keys here
   ```
4. Streamlit will automatically install the exact versions specified in `requirements.txt` (scikit-learn 1.8.0, xgboost 3.2.0) ensuring model compatibility and avoiding `_loss` module errors.

## Retraining the Model

If you receive new data (updated CSVs in the `data/` folder), you can retrain the model and regenerate all insights by simply running:

```bash
python model.py
```

This will automatically overwrite the `.pkl` model inside the `output/` folder. Reload the Streamlit app to reflect the new predictions.