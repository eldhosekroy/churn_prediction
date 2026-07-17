# ChurnSense AI – Candidate Analytics

ChurnSense AI is an AI-powered Candidate Churn Prediction & Reason Analysis platform. It leverages machine learning to proactively identify candidates at risk of dropping out (churning) during or before their training/induction sessions. 


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
   - **Role-Based Access Control (RBAC)**: Secure multi-tier authentication allowing *Admins* to see global analytics and *Salespersons* to access their focused workspaces.
   - **Salesperson Analytics & Overview**: Dedicated views for admins to track recruiter performance, churn risks per executive, and live system inferences.
   - **Smart Agent Workspace**: A focused, prioritized task manager for sales executives to track follow-ups and log interactions.
   - **Candidate Explorer & Live Predictor**: Deep-dive into individual candidate risk levels, input scenarios dynamically, and receive real-time churn probability and suggested actions.
   - **Mobile Responsive Design**: Clean, adaptive UI elements ensuring the dashboard is perfectly usable on smartphones and tablets.
   - **Supabase Integration**: Automatically logs queries, candidates, interaction history, and call insights into a Supabase PostgreSQL database.

2. **Machine Learning Pipeline (`model.py` & `prediction_models.py`)**:
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