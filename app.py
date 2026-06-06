"""
AI-Powered Candidate Churn Prediction & Reason Analysis Dashboard
=================================================================
Author: Dashboard Team Member
This file is NEW - it does NOT modify any existing team files.
It reads: Candidate Profile.csv, Call log.csv, Executive Profile.csv, churn_prediction_model.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv

import google.generativeai as genai
from google.ai import generativelanguage_v1beta as gal
import json
import os

# Groq and Hugging Face imports for LLM-based churn reason extraction
try:
    from groq import Groq

    groq_available = True
except ImportError:
    groq_available = False
    print("Warning: Groq is not available. Install with: pip install groq")

try:
    from huggingface_hub import inference

    huggingface_available = True
except ImportError:
    huggingface_available = False
    print("Warning: Hugging Face is not available. Install with: pip install huggingface_hub")


from supabase import create_client, Client

load_dotenv(override=True)

url: str = os.environ.get("SUPABASE_URL", "")
key: str = os.environ.get("SUPABASE_KEY", "")

supabase: Client | None = None
if url and key and url != "your-supabase-url":
    try:
        supabase = create_client(url, key)
        if "access_token" in st.session_state and "refresh_token" in st.session_state:
            try:
                supabase.auth.set_session(st.session_state.access_token, st.session_state.refresh_token)
            except Exception:
                pass
    except Exception as e:
        print(f"Failed to initialize Supabase: {e}")


def parse_gemini_response(response_data):
    if not isinstance(response_data, dict):
        return None

    text = None
    if 'output' in response_data:
        output = response_data['output']
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                content = first.get('content', first)
            else:
                content = first
            if isinstance(content, list):
                text = ''.join(item.get('text', '') if isinstance(item, dict) else str(item) for item in content)
            else:
                text = str(content)
    elif 'choices' in response_data:
        choices = response_data['choices']
        if isinstance(choices, list) and choices:
            choice = choices[0]
            text = choice.get('message', {}).get('content') or choice.get('text')
    return text


def parse_json_like(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    result = {}
    for line in text.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip().lower()] = value.strip()
    return result if result else None


def normalize_reason_label(text):
    if not text:
        return None
    normalized = text.strip().lower()
    mappings = {
        'Financial issues': ['financial issues', 'financial', 'payment', 'pay', 'fee', 'emi', 'installment', 'finance'],
        'Lack of interest': ['lack of interest', 'not interested', 'no interest', 'lost interest', 'not keen', 'disinterested', 'no longer interested'],
        'Joined another institution': ['joined another', 'joined other', 'admission elsewhere', 'admitted', 'migrated to', 'joined institute', 'joined company', 'enrolled elsewhere'],
        'Communication gaps': ['communication gaps', 'no response', 'no pickup', 'unreachable', 'voicemail', 'did not pick', 'not reachable', 'no answer', 'call dropped', 'busy', 'no contact', 'not responding'],
        'Other': ['other', 'unknown', 'unclear']
    }

    for label, keywords in mappings.items():
        if any(k in normalized for k in keywords):
            return label

    normalized_single = normalized.replace('\n', ' ').strip()
    return normalized_single.title() if normalized_single else 'Other'


def build_gemini_prompt(candidate_info, remarks_text, feedback_text, transcript_text=None):
    details = []
    if isinstance(candidate_info, dict):
        details.append("Candidate details:")
        for key in ['Source', 'Education', 'Background', 'Role', 'Current_status', 'Stream', 'Course', 'Mode', 'Payment_Method', 'Executive_Team', 'Induction_Session', 'Experience', 'Career_gap', 'Total_Amount', 'Paid_amount', 'Payment_Ratio', 'Zero_Payment', 'Negative_Feedback', 'High_Risk_Indicator', 'Days_Since_Payment', 'Total_Calls', 'Unique_Executives', 'Total_Call_Duration', 'Avg_Call_Duration', 'Max_Call_Duration', 'Min_Call_Duration', 'Call_Frequency', 'Executive_Experience', 'has_interest', 'has_no_response', 'has_payment_discussion', 'has_technical_discussion']:
            if key in candidate_info and candidate_info[key] is not None:
                details.append(f"- {key}: {candidate_info[key]}")
    else:
        details = ["Candidate details: Not available"]

    details.append("\nCall details:")
    details.append(f"- Feedback: {feedback_text or 'None'}")
    details.append(f"- Call remarks: {remarks_text or 'None'}")
    if transcript_text:
        details.append(f"- Call transcript: {transcript_text}")

    prompt = (
            "You are an expert AI candidate churn analyst for an IT professional training academy.\n"
            "Your task is to analyze candidate profile details and communication logs context to formulate a concise, logical, and personalized churn reason explanation alongside actionable recommendations.\n"
            "Value of the 'reason' key in the output JSON MUST be a comprehensive, detailed sentence or maximum three sentences explaining specifically why this candidate is churning, incorporating facts from their profile, payment details, and remarks.\n"
            "Value of the 'recommendation' key should be a highly logical, customized recovery plan based on their situation.\n\n"
            "Strict Format Constraint:\n"
            "You MUST respond ONLY with a clean JSON object containing exactly two keys: 'reason' and 'recommendation'. Do not include any standard prefixes, Markdown formatting blocks like ```json, or other notes. It must be clean, parsable JSON text.\n\n"
            "Candidate Data context:\n" + "\n".join(details)
    )
    return prompt


def call_gemini_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text=None):
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {'status': 'Gemini unavailable', 'error': 'API key missing'}

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    
    clean_model_name = model_name
    if clean_model_name.startswith('models/'):
        clean_model_name = clean_model_name[len('models/'):]

    # If a prohibited/legacy model is specified, fall back to the modern gemini-2.5-flash
    if any(m in clean_model_name.lower() for m in ['1.5-flash', '1.5-pro', 'gemini-pro', '2.0-flash', '2.0-pro']):
        clean_model_name = 'gemini-2.5-flash'

    prompt = build_gemini_prompt(candidate_info, remarks_text, feedback_text, transcript_text)

    import time
    last_err = None
    response_text = None

    # Implement exponential backoff retries to combat transient 503 errors
    for attempt in range(1, 4):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(clean_model_name)

            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            response_text = response.text
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(2 ** attempt)

    if last_err:
        return {'status': 'Fallback heuristic', 'error': str(last_err)}

    if not response_text:
        return {'status': 'Fallback heuristic', 'error': 'Empty response from model'}

    raw_text = response_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            raw_text = "\n".join(lines[1:-1])

    parsed = parse_json_like(raw_text)
    if parsed is None:
        parsed = {}

    # Extract detailed reason directly, do not map it into a 1-word label here
    reason = (parsed.get('reason') or parsed.get('reason_label') or raw_text).strip()
    recommendation = parsed.get('recommendation') or parsed.get('action') or ''
    return {'status': 'AI (Gemini)', 'reason': reason, 'recommendation': recommendation}

def heuristic_recommendation(reason_label):
    mapping = {
        'Financial issues': 'Offer flexible payment plans, scholarships, or budget-friendly EMI options and follow up on affordability concerns.',
        'Lack of interest': 'Re-engage with personalized course benefits, clarify learning outcomes, and offer a second consultation call.',
        'Joined another institution': 'Reach out with retention incentives, compare program strengths, and propose a unique value-added offer.',
        'Communication gaps': 'Increase outreach frequency, confirm contact details, and assign a dedicated counselor for follow-up.',
        'Other': 'Investigate the candidate details further and provide a customized recovery plan based on the latest call context.'
    }
    return mapping.get(reason_label, mapping['Other'])

def call_groq_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text=None):
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        return {'status': 'Groq unavailable', 'error': 'Groq API key missing'}
    try:
        from groq import Groq
    except ImportError:
        return {'status': 'Groq unavailable', 'error': 'groq package not installed'}

    try:
        prompt = build_gemini_prompt(candidate_info, remarks_text, feedback_text, transcript_text)
        client = Groq(api_key=groq_api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert HR analyst analyzing candidate churn data. Respond ONLY with a clean JSON object containing exactly two keys: 'reason' and 'recommendation'."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=256
        )
        response_text = chat_completion.choices[0].message.content.strip()
        raw_text = response_text
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                raw_text = "\n".join(lines[1:-1])

        parsed = parse_json_like(raw_text)
        if not parsed:
            parsed = {}
        reason = (parsed.get('reason') or parsed.get('reason_label') or raw_text).strip()
        recommendation = parsed.get('recommendation') or parsed.get('action') or ''
        return {'status': 'AI (Groq)', 'reason': reason, 'recommendation': recommendation}
    except Exception as e:
        return {'status': 'Groq failed', 'error': str(e)}


def call_huggingface_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text=None):
    hf_api_key = os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HF_TOKEN')
    if not hf_api_key:
        return {'status': 'HuggingFace unavailable', 'error': 'Hugging Face API key missing'}
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return {'status': 'HuggingFace unavailable', 'error': 'huggingface_hub package not installed'}

    try:
        prompt = build_gemini_prompt(candidate_info, remarks_text, feedback_text, transcript_text)
        client = InferenceClient(api_key=hf_api_key)
        response = client.text_generation(
            prompt=prompt + "\nJSON output:",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=256,
            temperature=0.3
        )
        response_text = response.strip()
        raw_text = response_text
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                raw_text = "\n".join(lines[1:-1])

        parsed = parse_json_like(raw_text)
        if not parsed:
            parsed = {}
        reason = (parsed.get('reason') or parsed.get('reason_label') or raw_text).strip()
        recommendation = parsed.get('recommendation') or parsed.get('action') or ''
        return {'status': 'AI (Hugging Face)', 'reason': reason, 'recommendation': recommendation}
    except Exception as e:
        return {'status': 'HuggingFace failed', 'error': str(e)}

def extract_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text=None):
    api_response = call_gemini_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text)
    if api_response and api_response.get('status') == 'AI (Gemini)' and api_response.get('reason'):
        return api_response['reason'], api_response.get('recommendation', ''), api_response['status']

    errors = []
    if api_response and 'error' in api_response:
        errors.append(f"Gemini: {api_response['error']}")

    groq_response = call_groq_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text)
    if groq_response and groq_response.get('status') == 'AI (Groq)' and groq_response.get('reason'):
        return groq_response['reason'], groq_response.get('recommendation', ''), groq_response['status']

    if groq_response and 'error' in groq_response:
        errors.append(f"Groq: {groq_response['error']}")

    hf_response = call_huggingface_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text)
    if hf_response and hf_response.get('status') == 'AI (Hugging Face)' and hf_response.get('reason'):
        return hf_response['reason'], hf_response.get('recommendation', ''), hf_response['status']

    if hf_response and 'error' in hf_response:
        errors.append(f"Hugging Face: {hf_response['error']}")

    error_context = " | ".join(errors) if errors else 'Unknown AI failure'
    text = ''
    if remarks_text:
        text += str(remarks_text).lower() + ' '
    if feedback_text:
        text += str(feedback_text).lower()

    if any(k in text for k in ['pay', 'payment', 'fee', 'installment', 'emi', 'finance', 'financial']):
         reason = 'Financial issues: Candidate is flagged for high churn risk due to fee, outstanding payment, or EMI installment concerns mentioned in call log details.'
         label_key = 'Financial issues'
    elif any(k in text for k in ['not interested', 'no interest', 'lack of interest', 'lost interest', 'not keen', 'disinterested', 'no longer interested']):
         reason = 'Lack of interest: Candidate exhibits disinterest, program mismatch, or lack of direct engagement with onboarding tasks.'
         label_key = 'Lack of interest'
    elif any(k in text for k in ['joined another', 'joined other', 'admission elsewhere', 'admitted', 'migrated to', 'joined institute', 'joined company', 'enrolled elsewhere']):
         reason = 'Joined another institution: Candidate explicitly opted for admission or alternative training outcomes at another institution.'
         label_key = 'Joined another institution'
    elif any(k in text for k in ['no response', 'no pickup', 'unreachable', 'voicemail', 'did not pick', 'not reachable', 'no answer', 'call dropped', 'busy', 'no contact', 'not responding']):
         reason = 'Communication gaps: Candidate has a high rate of unreachability, busy signals, or unanswered outbound contact attempts.'
         label_key = 'Communication gaps'
    elif any(k in text for k in ['course not suitable', 'course mismatch', 'course not for me', 'content not relevant']):
         reason = 'Lack of interest: Candidate exhibits disinterest, program mismatch, or lack of direct engagement with onboarding tasks.'
         label_key = 'Lack of interest'
    else:
         reason = 'Other: Candidate exhibits general warning indicators or ambiguous communication feedback requiring dedicated outreach.'
         label_key = 'Other'

    return reason, heuristic_recommendation(label_key), f'Fallback heuristic ({error_context})'


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense AI – Candidate Analytics",
    page_icon="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/bullseye.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS – Premium Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: #0f1115;
    }

    /* Hide default streamlit header and reduce top padding */
    #MainMenu, footer { visibility: hidden; }
    header { background-color: transparent !important; }
    .stDeployButton, .stAppDeployButton { display: none !important; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Left align sidebar buttons to look like nav links and reduce vertical gap */
    [data-testid="stSidebar"] .stButton {
        margin-bottom: -12px;
    }
    [data-testid="stSidebar"] button {
        justify-content: flex-start !important;
        padding-left: 16px !important;
        background-color: transparent !important;
        border-color: transparent !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        border-radius: 4px 8px 8px 4px !important;
        border-left: 3px solid transparent !important;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255,255,255,0.05) !important;
        color: #ffffff !important;
    }

    /* Hide input instructions (Press Enter to apply) */
    [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Specifically style the Log Out button (last button in sidebar) */
    [data-testid="stSidebar"] .stButton:last-of-type button {
        background-color: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        margin-top: 10px !important;
    }
    [data-testid="stSidebar"] .stButton:last-of-type button:hover {
        background-color: rgba(239,68,68,0.1) !important;
        border-color: rgba(239,68,68,0.3) !important;
        color: #f87171 !important;
    }

    /* KPI Cards */
    .kpi-card {
        background: #1a1d24;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 24px 20px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 8px;
        height: 190px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 36px;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 8px;
    }
    .kpi-sub {
        font-size: 13px;
        color: #64748b;
    }
    .kpi-icon { font-size: 24px; margin-bottom: 12px; color: #475569; }
    .kpi-red .kpi-value { color: #f87171; }
    .kpi-green .kpi-value { color: #34d399; }
    .kpi-blue .kpi-value { color: #60a5fa; }
    .kpi-amber .kpi-value { color: #fbbf24; }

    /* Section headers */
    .section-header {
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding: 0 0 8px 0;
        margin: 32px 0 16px 0;
    }
    .section-header h2 {
        color: #e2e8f0;
        font-size: 16px;
        font-weight: 600;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Page header */
    .page-header {
        padding: 0 0 24px 0;
        margin-bottom: 32px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .page-header h1 {
        color: #ffffff;
        font-size: 32px;
        font-weight: 600;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .page-header p {
        color: #94a3b8;
        margin: 0;
        font-size: 15px;
    }

    /* Risk badges */
    .badge-high { background: rgba(239,68,68,0.1); color: #f87171; border: 1px solid rgba(239,68,68,0.2); padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge-medium { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge-low { background: rgba(52,211,153,0.1); color: #34d399; border: 1px solid rgba(52,211,153,0.2); padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; }

    /* Candidate card */
    .candidate-card {
        background: #1a1d24;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        gap: 24px;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 500;
        padding: 12px 4px;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: transparent !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Prediction box */
    .prediction-box-churn {
        background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.1) 100%);
        border: 2px solid rgba(239,68,68,0.5);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
    }
    .prediction-box-safe {
        background: linear-gradient(135deg, rgba(52,211,153,0.2) 0%, rgba(16,185,129,0.1) 100%);
        border: 2px solid rgba(52,211,153,0.5);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
    }
    .pred-label {
        font-size: 48px;
        font-weight: 900;
        margin-bottom: 8px;
    }
    .pred-sub {
        color: #94a3b8;
        font-size: 15px;
    }

    /* Sidebar nav button style */
    .nav-item {
        padding: 10px 16px;
        border-radius: 10px;
        margin: 4px 0;
        cursor: pointer;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.2s;
    }
    .nav-item.active, .nav-item:hover {
        background: rgba(255,255,255,0.2);
        color: #a78bfa;
    }

    /* Input fields */
    .stSelectbox label p, .stNumberInput label p, .stTextInput label p {
        color: #a78bfa !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .stSelectbox > div, .stNumberInput > div, .stTextInput > div {
        background: rgba(15, 12, 41, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Metric delta */
    [data-testid="stMetricDelta"] { font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# (Replicates model.py logic — files untouched)
# ─────────────────────────────────────────────

INPUT_DIR = "./data/"
OUTPUT_DIR = "./outputs/"

@st.cache_data
def load_data():
    candidate_profile = pd.read_csv(INPUT_DIR + "Candidate Profile.csv")
    call_log          = pd.read_csv(INPUT_DIR + "Call log.csv")
    executive_profile = pd.read_csv(INPUT_DIR + "Executive Profile.csv")
    return candidate_profile, call_log, executive_profile


@st.cache_data
def load_churn_reasons():
    """Attempt to load churn reason outputs saved by model.py. Returns full and short dataframes or (None, None)."""
    full_path = os.path.join(OUTPUT_DIR, 'candidates_with_suggested_reasons.csv')
    short_path = os.path.join(OUTPUT_DIR, 'churn_reasons.csv')
    full_df = None
    short_df = None
    try:
        if os.path.exists(full_path):
            full_df = pd.read_csv(full_path)
    except Exception:
        full_df = None
    try:
        if os.path.exists(short_path):
            short_df = pd.read_csv(short_path)
    except Exception:
        short_df = None
    return full_df, short_df


@st.cache_data
def preprocess(candidate_profile, call_log, executive_profile):
    # ── Churn Label ──────────────────────────────
    def define_churn(row):
        training_incomplete = ['not started', 'pending', 'incomplete', 'not completed', 'notjoined']
        if pd.notna(row['Training_Session']):
            if any(s in str(row['Training_Session']).lower() for s in training_incomplete):
                return 1
        return 0

    df_candidate = candidate_profile.copy()
    df_candidate['Churn'] = df_candidate.apply(define_churn, axis=1)

    # ── Missing Values ───────────────────────────
    numerical_cols  = df_candidate.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols  = [c for c in numerical_cols if c not in ['Candidate_ID', 'Churn']]
    categorical_cols= df_candidate.select_dtypes(include=['object']).columns.tolist()
    categorical_cols= [c for c in categorical_cols if c not in ['Candidate_ID', 'Mail_ID', 'Mobile_Number']]

    for col in numerical_cols:
        df_candidate[col].fillna(df_candidate[col].median(), inplace=True)
    for col in categorical_cols:
        df_candidate[col].fillna(
            df_candidate[col].mode()[0] if len(df_candidate[col].mode()) > 0 else 'Unknown',
            inplace=True
        )

    # ── Date Features ────────────────────────────
    df_candidate['Payment_Date'] = pd.to_datetime(df_candidate['Payment_Date'], errors='coerce')
    reference_date = pd.Timestamp.today().normalize()
    df_candidate['Days_Since_Payment'] = (reference_date - df_candidate['Payment_Date']).dt.days
    df_candidate['Days_Since_Payment'].fillna(0, inplace=True)

    # ── Payment Ratio ────────────────────────────
    df_candidate['Payment_Ratio'] = np.where(
        df_candidate['Total_Amount'] > 0,
        df_candidate['Paid_amount'] / df_candidate['Total_Amount'],
        0
    )
    df_candidate['Payment_Ratio'] = df_candidate['Payment_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    df_candidate['Outstanding_Amount'] = df_candidate['Total_Amount'] - df_candidate['Paid_amount']
    df_candidate['Booking_fee'] = (df_candidate['Paid_amount'] == 2000).astype(int)
    df_candidate['Negative_Feedback'] = df_candidate['Feedback'].astype(str).str.strip().str.lower().eq('negative').astype(int)

    # ── Call Log Processing ──────────────────────
    call_log_proc = call_log.copy()
    def to_minutes(d):
        if pd.isna(d): return 0
        parts = str(d).split(':')
        return round(int(parts[0]) + (int(parts[1]) if len(parts) > 1 else 0) / 60, 2)

    call_log_proc['Call_Duration'] = call_log_proc['Call_Duration'].apply(to_minutes)
    call_log_proc['Call_Date']     = pd.to_datetime(call_log_proc['Call_Date'], errors='coerce')

    call_agg = call_log_proc.groupby('Candidate_ID').agg(
        Total_Calls       = ('Executive_ID', 'count'),
        Unique_Executives = ('Executive_ID', 'nunique'),
        Total_Call_Duration = ('Call_Duration', 'sum'),
        Avg_Call_Duration   = ('Call_Duration', 'mean'),
        Max_Call_Duration   = ('Call_Duration', 'max'),
        Min_Call_Duration   = ('Call_Duration', 'min'),
    ).reset_index()

    date_range = max((call_log_proc['Call_Date'].max() - call_log_proc['Call_Date'].min()).days, 1)
    call_counts = call_log_proc.groupby('Candidate_ID').size().reset_index(name='Call_Count')
    call_counts['Call_Frequency'] = call_counts['Call_Count'] / (date_range / 30)
    call_agg = call_agg.merge(call_counts, on='Candidate_ID', how='left')
    call_agg['Call_Frequency'].fillna(0, inplace=True)

    # ── Remark NLP Features ──────────────────────
    remark_list = []
    for cid in call_log_proc['Candidate_ID'].unique():
        sub = call_log_proc[call_log_proc['Candidate_ID'] == cid]['Call_Remarks'].astype(str).str.lower()
        remark_list.append({
            'Candidate_ID': cid,
            'has_interest':            int(sub.str.contains('interested|keen|enthusiastic|confirmed|enrolled').any()),
            'has_no_interest':         int(sub.str.contains('not interested|lack of interest|no confirmation|not enrolled').any()),
            'has_no_response':         int(sub.str.contains('no response|no pickup|unreachable|voicemail').any()),
            'joined_another':          int(sub.str.contains('joined another instituition|another|instituition').any()),
            'has_payment_discussion':  int(sub.str.contains('payment|fee|emi|scholarship').any()),
            'has_technical_discussion':int(sub.str.contains('technical|syllabus|project|mentor').any()),
        })
    remark_df = pd.DataFrame(remark_list)
    call_agg  = call_agg.merge(remark_df, on='Candidate_ID', how='left').fillna(0)

    # ── Executive Features ───────────────────────
    exec_feat = executive_profile[['Executive_ID', 'Experience_Years', 'Team']].copy()
    exec_feat.rename(columns={'Experience_Years':'Executive_Experience',
                               'Team':'Executive_Team'}, inplace=True)
    call_with_exec = call_log_proc.merge(exec_feat, on='Executive_ID', how='left')
    exec_agg = call_with_exec.groupby('Candidate_ID').agg(
        Executive_Experience = ('Executive_Experience', 'mean'),
        Executive_Team       = ('Executive_Team',       lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'),
    ).reset_index()
    call_agg = call_agg.merge(exec_agg, on='Candidate_ID', how='left')

    # ── Merge All ────────────────────────────────
    df = df_candidate.merge(call_agg, on='Candidate_ID', how='left')
    num_fill = {c: 0 for c in call_agg.columns if c != 'Candidate_ID' and df[c].dtype in [np.float64, np.int64]}
    df.fillna(num_fill, inplace=True)
    if 'Executive_Team' in df.columns:
        df['Executive_Team'].fillna('No Contact', inplace=True)

    return df, call_log_proc


@st.cache_resource
def load_model(model_path, modified_time):
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            if 'model_name' not in data and 'model' in data:
                data['model_name'] = data['model'].__class__.__name__
            if 'model_display_name' not in data:
                data['model_display_name'] = data.get('model_name', data['model'].__class__.__name__)
            if 'balance_method' not in data:
                data['balance_method'] = 'none'
            return data
    except Exception:
        return None


def format_balance_method(balance_method):
    balance_labels = {
        'none': 'None',
        'class_weight': 'Class Weight',
        'oversample': 'Random Oversampling',
        'undersample': 'Random Undersampling',
        'smote': 'SMOTE'
    }
    return balance_labels.get(str(balance_method).lower(), str(balance_method))


def balance_method_description(balance_method):
    descriptions = {
        'none': 'Training used the original class distribution.',
        'class_weight': 'The model gave more weight to the minority churn class during training.',
        'oversample': 'The minority churn class was randomly oversampled before training.',
        'undersample': 'The majority class was randomly undersampled before training.',
        'smote': 'Synthetic minority samples were generated with SMOTE before training.'
    }
    return descriptions.get(str(balance_method).lower(), 'Selected from validation F1 during model training.')


# ─────────────────────────────────────────────
# PLOTLY THEME DEFAULTS
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False),
    margin=dict(l=20, r=20, t=40, b=20),
)

# Default legend style applied everywhere
_LEGEND_DEFAULTS = dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))

def theme(**overrides):
    """Return PLOTLY_THEME merged with any per-chart overrides.
    Handles 'legend' specially so callers can pass legend=dict(...)
    without a duplicate-keyword error."""
    merged = dict(PLOTLY_THEME)
    if 'legend' in overrides:
        leg = dict(_LEGEND_DEFAULTS)   # start with defaults
        leg.update(overrides.pop('legend'))  # layer caller's settings on top
        merged['legend'] = leg
    else:
        merged['legend'] = _LEGEND_DEFAULTS
    merged.update(overrides)
    return merged
COLOR_ACTIVE = '#34d399'
COLOR_CHURN  = '#f87171'
PALETTE      = ['#6366f1','#8b5cf6','#06b6d4','#f59e0b','#10b981','#ef4444','#3b82f6','#ec4899','#f97316','#84cc16']


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def page_auth():
    st.markdown("<div style='text-align:center; padding:50px 0;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 42px; font-weight: 700; letter-spacing: 0.5px; line-height: 1.1;">
            <span style="color: #f8fafc;">Churn</span><span style="background: -webkit-linear-gradient(45deg, #a78bfa, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Sense</span>
            <span style="font-family: 'Inter', sans-serif; font-size: 15px; font-weight: 900; letter-spacing: 1px; color: #38bdf8; vertical-align: top; margin-left: 2px;">AI</span>
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #94a3b8; margin-top: 10px; text-transform: uppercase; letter-spacing: 3.5px; font-weight: 500;">
            Executive Portal
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if supabase is None:
        st.error("Supabase credentials not configured. Please check your `.env` file.", icon=":material/warning:")
        st.stop()

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <i class="fa-solid fa-lock" style="font-size: 32px; color: #38bdf8; margin-bottom: 16px;"></i>
                <h1 style="font-family: 'Playfair Display', serif; font-size: 42px; margin: 0; color: #f8fafc; font-weight: 700;">Executive Login</h1>
                <p style="color: #94a3b8; font-size: 14px; margin-top: 8px;">Access the secure analytics dashboard</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form", border=False):
                login_email = st.text_input("Email Address", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                if submitted:
                    if login_email and login_password:
                        try:
                            res = supabase.auth.sign_in_with_password({"email": login_email, "password": login_password})
                            st.session_state.logged_in = True
                            st.session_state.user_email = res.user.email
                            if res.session:
                                st.session_state.access_token = res.session.access_token
                                st.session_state.refresh_token = res.session.refresh_token
                            st.rerun()
                        except Exception as e:
                            st.error(f"Login failed: {e}")
                    else:
                        st.warning("Please enter email and password.")

        with tab2:
            st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <i class="fa-solid fa-user-plus" style="font-size: 32px; color: #34d399; margin-bottom: 16px;"></i>
                <h1 style="font-family: 'Playfair Display', serif; font-size: 42px; margin: 0; color: #f8fafc; font-weight: 700;">Create Account</h1>
                <p style="color: #94a3b8; font-size: 14px; margin-top: 8px;">Register for executive access</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("register_form", border=False):
                reg_email = st.text_input("Email Address", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
                submitted_reg = st.form_submit_button("Sign Up", type="primary", use_container_width=True)
                if submitted_reg:
                    if reg_email and reg_password:
                        try:
                            res = supabase.auth.sign_up({"email": reg_email, "password": reg_password})
                            st.success("Registration successful! You can now log in using the Login tab.")
                        except Exception as e:
                            st.error(f"Registration failed: {e}")
                    else:
                        st.warning("Please enter email and password.")


def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 25px 0 20px 0; background: linear-gradient(180deg, rgba(30,41,59,0.4) 0%, transparent 100%); border-bottom: 1px solid rgba(255,255,255,0.03); margin-bottom: 15px;">
            <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 32px; font-weight: 700; letter-spacing: 0.5px; line-height: 1.1;">
                <span style="color: #f8fafc;">Churn</span><span style="background: -webkit-linear-gradient(45deg, #a78bfa, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Sense</span>
                <span style="font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 900; letter-spacing: 1px; color: #38bdf8; vertical-align: top; margin-left: 2px;">AI</span>
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 10px; color: #94a3b8; margin-top: 10px; text-transform: uppercase; letter-spacing: 3.5px; font-weight: 500;">
                Candidate Analytics
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "current_page" not in st.session_state:
            st.session_state.current_page = "Overview"

        pages = [
            ("Overview", ":material/dashboard:"),
            ("Candidate Explorer", ":material/search:"),
            ("Call Log Analysis", ":material/call:"),
            ("Payment Analysis", ":material/payments:"),
            ("Live Predictor", ":material/online_prediction:"),
            ("Model Performance", ":material/insights:")
        ]

        for p_name, p_icon in pages:
            if st.button(p_name, icon=p_icon, use_container_width=True):
                st.session_state.current_page = p_name
                st.session_state.show_profile = False
                st.rerun()

        # Dynamically inject CSS to highlight the active button by its exact DOM index
        active_idx = [p[0] for p in pages].index(st.session_state.current_page) + 2
        st.markdown(f"""
        <style>
            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div.element-container:nth-child({active_idx}) button {{
                font-weight: 600 !important;
                background-color: rgba(56,189,248,0.1) !important;
                color: #38bdf8 !important;
                border-left: 3px solid #38bdf8 !important;
            }}
        </style>
        """, unsafe_allow_html=True)

        page = st.session_state.current_page

        st.markdown("<hr style='border-color:rgba(255,255,255,0.2);margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:10px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;margin:0 0 8px 0;'>Data Sources</p>", unsafe_allow_html=True)

        _src = [
            ("<i class='fa-solid fa-clipboard-list'></i>", "Candidate Profile", ".csv &nbsp;·&nbsp; 50 rows"),
            ("<i class='fa-solid fa-phone'></i>", "Call Log",           ".csv &nbsp;·&nbsp; 124 rows"),
            ("<i class='fa-solid fa-user-tie'></i>", "Executive Profile",  ".csv &nbsp;·&nbsp; 10 rows"),
            ("<i class='fa-solid fa-robot'></i>", "Churn Model",        ".pkl &nbsp;·&nbsp; Saved Model"),
        ]
        for _icon, _name, _meta in _src:
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.18);
            border-radius:9px;padding:8px 10px;margin-bottom:6px;overflow:hidden;">
  <span style="font-size:15px;vertical-align:middle;">{_icon}</span>
  <span style="font-size:11px;font-weight:600;color:#cbd5e1;
               vertical-align:middle;margin-left:8px;">{_name}</span>
  <span style="float:right;width:7px;height:7px;border-radius:50%;
               background:#34d399;display:inline-block;margin-top:5px;"></span>
  <div style="font-size:10px;color:#475569;margin-top:3px;padding-left:28px;">{_meta}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.2);
            border-radius:9px;padding:8px 12px;margin-top:4px;overflow:hidden;">
  <span style="font-size:13px;vertical-align:middle;"><i class="fa-solid fa-tools"></i></span>
  <span style="font-size:10px;color:#64748b;vertical-align:middle;margin-left:6px;">
    <b style="color:#34d399;">Read-only</b> &mdash; no team files modified
  </span>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        if st.button("Log Out", icon=":material/logout:", type="secondary", use_container_width=True):
            st.session_state.logged_in = False
            if "access_token" in st.session_state:
                del st.session_state["access_token"]
            if "refresh_token" in st.session_state:
                del st.session_state["refresh_token"]
            try:
                supabase.auth.sign_out()
            except:
                pass
            st.rerun()

    return page


# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
def page_overview(df, call_log_proc, churn_full=None):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-chart-pie"></i> Executive Overview</h1>
        <p>Real-time snapshot of candidate churn status, sources, and engagement metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    total       = len(df)
    churned     = df['Churn'].sum()
    active      = total - churned
    churn_rate  = churned / total * 100
    avg_payment = df['Payment_Ratio'].mean() * 100
    total_calls = len(call_log_proc)

    # ── KPI Cards ────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon"><i class="fa-solid fa-users"></i></div>
            <div class="kpi-title">Total Candidates</div>
            <div class="kpi-value kpi-blue" style="color:#60a5fa">{total}</div>
            <div class="kpi-sub">Enrolled in system</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon"><i class="fa-solid fa-check-circle" style="color:#34d399"></i></div>
            <div class="kpi-title">Active</div>
            <div class="kpi-value" style="color:#34d399">{active}</div>
            <div class="kpi-sub">Training joined</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon"><i class="fa-solid fa-triangle-exclamation" style="color:#f87171"></i></div>
            <div class="kpi-title">Churned</div>
            <div class="kpi-value" style="color:#f87171">{int(churned)}</div>
            <div class="kpi-sub">Did not join training</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon"><i class="fa-solid fa-chart-line" style="color:#fbbf24"></i></div>
            <div class="kpi-title">Churn Rate</div>
            <div class="kpi-value" style="color:#fbbf24">{churn_rate:.1f}%</div>
            <div class="kpi-sub">Of all candidates</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon"><i class="fa-solid fa-phone" style="color:#a78bfa"></i></div>
            <div class="kpi-title">Total Calls</div>
            <div class="kpi-value" style="color:#a78bfa">{total_calls}</div>
            <div class="kpi-sub">Across all candidates</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Suggested Churn Reasons (from model outputs) ─────────
    if churn_full is not None and 'Suggested_Churn_Reason' in churn_full.columns:
        try:
            mapped_reasons = churn_full[churn_full['Churn'] == 1]['Suggested_Churn_Reason'].dropna().astype(str).apply(normalize_reason_label)
            reasons = mapped_reasons.value_counts().reset_index()
            reasons.columns = ['Reason', 'Count']

            if not reasons.empty:
                st.markdown('<div class="section-header"><h2>Top Suggested Churn Reasons</h2></div>', unsafe_allow_html=True)
                fig_reasons = px.bar(reasons, x='Count', y='Reason', orientation='h', text='Count', color='Count', color_continuous_scale=['#f87171','#fbbf24','#60a5fa'])
                fig_reasons.update_layout(**theme(height=300, showlegend=False))
                st.plotly_chart(fig_reasons, use_container_width=True)
        except Exception:
            pass

    # ── Row 1: Donut + Source ─────────────────────
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="section-header"><h2>Churn Distribution</h2></div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Active', 'Churned'],
            values=[active, int(churned)],
            hole=0.65,
            marker=dict(colors=[COLOR_ACTIVE, COLOR_CHURN],
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            textinfo='percent+label',
            textfont=dict(size=13, color='#e2e8f0'),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        ))
        fig.add_annotation(text=f"<b>{churn_rate:.1f}%</b><br>Churn Rate",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=15, color='#e2e8f0', family='Inter'))
        fig.update_layout(**theme(height=310,
                                  showlegend=True,
                                  legend=dict(orientation='h', y=-0.1, x=0.25)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h2>Churn by Candidate Source</h2></div>', unsafe_allow_html=True)
        src = df.groupby(['Source', 'Churn']).size().reset_index(name='Count')
        src['Status'] = src['Churn'].map({0: 'Active', 1: 'Churned'})
        fig2 = px.bar(src, x='Source', y='Count', color='Status',
                      color_discrete_map={'Active': COLOR_ACTIVE, 'Churned': COLOR_CHURN},
                      barmode='group', text='Count')
        fig2.update_traces(textfont_size=11, textposition='outside')
        fig2.update_layout(**theme(height=310, showlegend=True))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Course + Background + Mode ─────────
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown('<div class="section-header"><h2>By Course</h2></div>', unsafe_allow_html=True)
        course_churn = df[df['Churn'] == 1]['Course'].value_counts().reset_index()
        course_churn.columns = ['Course', 'Churned']
        fig3 = px.bar(course_churn, x='Churned', y='Course', orientation='h',
                      color='Churned', color_continuous_scale=['#4c1d95','#f87171'])
        fig3.update_layout(**theme(height=280, showlegend=False,
                                   coloraxis_showscale=False))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header"><h2>Background Split</h2></div>', unsafe_allow_html=True)
        bg = df.groupby(['Background', 'Churn']).size().reset_index(name='Count')
        bg['Status'] = bg['Churn'].map({0: 'Active', 1: 'Churned'})
        fig4 = px.pie(bg, names='Background', values='Count',
                      color='Background', hole=0.5,
                      color_discrete_sequence=PALETTE)
        fig4.update_layout(**theme(height=280, showlegend=True))
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.markdown('<div class="section-header"><h2>Training Mode</h2></div>', unsafe_allow_html=True)
        mode_data = df.groupby(['Mode', 'Churn']).size().reset_index(name='Count')
        mode_data['Status'] = mode_data['Churn'].map({0: 'Active', 1: 'Churned'})
        fig5 = px.bar(mode_data, x='Mode', y='Count', color='Status',
                      color_discrete_map={'Active': COLOR_ACTIVE, 'Churned': COLOR_CHURN},
                      barmode='stack')
        fig5.update_layout(**theme(height=280))
        st.plotly_chart(fig5, use_container_width=True)

    # ── Row 3: Induction + Feedback ───────────────
    col6, col7 = st.columns(2)
    with col6:
        st.markdown('<div class="section-header"><h2>Induction Attendance vs Churn</h2></div>', unsafe_allow_html=True)
        ind = df.groupby(['Induction_Session', 'Churn']).size().reset_index(name='Count')
        ind['Status'] = ind['Churn'].map({0: 'Active', 1: 'Churned'})
        fig6 = px.bar(ind, x='Induction_Session', y='Count', color='Status',
                      color_discrete_map={'Active': COLOR_ACTIVE, 'Churned': COLOR_CHURN},
                      barmode='group', text='Count')
        fig6.update_traces(textfont_size=11, textposition='outside')
        fig6.update_layout(**theme(height=290))
        st.plotly_chart(fig6, use_container_width=True)

    with col7:
        st.markdown('<div class="section-header"><h2>Candidate Feedback vs Churn</h2></div>', unsafe_allow_html=True)
        fb = df.groupby(['Feedback', 'Churn']).size().reset_index(name='Count')
        fb['Status'] = fb['Churn'].map({0: 'Active', 1: 'Churned'})
        fig7 = px.bar(fb, x='Feedback', y='Count', color='Status',
                      color_discrete_map={'Active': COLOR_ACTIVE, 'Churned': COLOR_CHURN},
                      barmode='group', text='Count')
        fig7.update_traces(textfont_size=11, textposition='outside')
        fig7.update_layout(**theme(height=290))
        st.plotly_chart(fig7, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2 — CANDIDATE EXPLORER
# ─────────────────────────────────────────────
def page_candidate_explorer(df, call_log_proc, executive_profile, churn_full=None):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-magnifying-glass"></i> Candidate Explorer</h1>
        <p>Browse, filter, and inspect individual candidate records with call history.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────
    with st.expander("Filter Candidates", expanded=True):
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            sel_churn = st.selectbox("Churn Status", ["All", "Churned", "Active"], key="f_churn")
        with f2:
            sel_source = st.selectbox("Source", ["All"] + sorted(df['Source'].unique().tolist()), key="f_source")
        with f3:
            sel_course = st.selectbox("Course", ["All"] + sorted(df['Course'].unique().tolist()), key="f_course")
        with f4:
            sel_mode = st.selectbox("Mode", ["All"] + sorted(df['Mode'].unique().tolist()), key="f_mode")
        with f5:
            sel_bg = st.selectbox("Background", ["All"] + sorted(df['Background'].unique().tolist()), key="f_bg")

    fdf = df.copy()

    # If churn suggestions exist, merge them into the working dataframe for display
    if churn_full is not None and 'Candidate_ID' in churn_full.columns and 'Suggested_Churn_Reason' in churn_full.columns:
        try:
            reason_map = churn_full.set_index('Candidate_ID')['Suggested_Churn_Reason'].to_dict()
            fdf['Suggested_Churn_Reason'] = fdf['Candidate_ID'].map(reason_map).fillna('')
        except Exception:
            fdf['Suggested_Churn_Reason'] = ''
    if sel_churn == "Churned": fdf = fdf[fdf['Churn'] == 1]
    elif sel_churn == "Active": fdf = fdf[fdf['Churn'] == 0]
    if sel_source != "All": fdf = fdf[fdf['Source'] == sel_source]
    if sel_course != "All": fdf = fdf[fdf['Course'] == sel_course]
    if sel_mode   != "All": fdf = fdf[fdf['Mode']   == sel_mode]
    if sel_bg     != "All": fdf = fdf[fdf['Background'] == sel_bg]

    st.markdown(f"<p style='color:#64748b; font-size:13px;'>Showing <b style='color:#a78bfa'>{len(fdf)}</b> candidates</p>", unsafe_allow_html=True)

    # ── Table ─────────────────────────────────────
    display_cols = ['Candidate_ID', 'Candidate_Name', 'Source', 'Course', 'Mode',
                    'Background', 'Role', 'Training_Session', 'Feedback',
                    'Total_Amount', 'Paid_amount', 'Payment_Ratio', 'Total_Calls', 'Churn', 'Suggested_Churn_Reason']
    display_cols = [c for c in display_cols if c in fdf.columns]

    tbl = fdf[display_cols].copy()
    tbl['Churn'] = tbl['Churn'].map({0: 'Active', 1: 'Churned'})
    if 'Payment_Ratio' in tbl.columns:
        tbl['Payment_Ratio'] = (tbl['Payment_Ratio'] * 100).round(1).astype(str) + '%'

    st.dataframe(
        tbl.reset_index(drop=True),
        use_container_width=True,
        height=280,
        column_config={
            "Candidate_ID":    st.column_config.TextColumn("ID", width="small"),
            "Candidate_Name":  st.column_config.TextColumn("Name"),
            "Training_Session":st.column_config.TextColumn("Training Status"),
            "Total_Amount":    st.column_config.NumberColumn("Total Fee (₹)", format="₹%d"),
            "Paid_amount":     st.column_config.NumberColumn("Paid (₹)", format="₹%d"),
            "Total_Calls":     st.column_config.NumberColumn("Calls"),
            "Churn":           st.column_config.TextColumn("Status"),
        }
    )

    # ── Individual Profile ────────────────────────
    st.markdown('<div class="section-header"><h2>Candidate Profile Deep-Dive</h2></div>', unsafe_allow_html=True)
    candidate_ids = fdf['Candidate_ID'].tolist()
    if not candidate_ids:
        st.info("No candidates match the current filters.")
        return

    sel_id = st.selectbox("Select Candidate", candidate_ids, key="profile_id")
    row    = df[df['Candidate_ID'] == sel_id].iloc[0]
    # If merged suggestions were added to fdf, prefer that for the selected row
    if 'Suggested_Churn_Reason' in fdf.columns:
        row = fdf[fdf['Candidate_ID'] == sel_id].iloc[0]
    calls  = call_log_proc[call_log_proc['Candidate_ID'] == sel_id].copy()

    churn_label = "CHURNED" if row['Churn'] == 1 else "ACTIVE"
    churn_color = "#f87171" if row['Churn'] == 1 else "#34d399"

    pc1, pc2, pc3 = st.columns([1.2, 1.2, 1.6])
    with pc1:
        st.markdown(f"""
        <div class="candidate-card">
            <div style="font-size:13px; font-weight:700; color:#94a3b8; margin-bottom:12px; text-transform:uppercase; letter-spacing:1px;">Personal Info</div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Name:</span> <b style="color:#e2e8f0;">{row.get('Candidate_Name','N/A')}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">ID:</span> <b style="color:#a78bfa;">{row['Candidate_ID']}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Background:</span> <b style="color:#e2e8f0;">{row.get('Background','N/A')}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Role:</span> <b style="color:#e2e8f0;">{row.get('Role','N/A')}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Experience:</span> <b style="color:#e2e8f0;">{row.get('Experience',0)} yrs</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Education:</span> <b style="color:#e2e8f0;">{row.get('Education','N/A')}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Source:</span> <b style="color:#e2e8f0;">{row.get('Source','N/A')}</b></div>
            <div style="margin-top:14px; padding:8px 14px; border-radius:8px; background:rgba(0,0,0,0.2); text-align:center;">
                <span style="color:{churn_color}; font-weight:800; font-size:16px;">{churn_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with pc2:
        pr = row.get('Payment_Ratio', 0) * 100
        st.markdown(f"""
        <div class="candidate-card">
            <div style="font-size:13px; font-weight:700; color:#94a3b8; margin-bottom:12px; text-transform:uppercase; letter-spacing:1px;">Course & Payment</div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Course:</span> <b style="color:#e2e8f0;">{row.get('Course','N/A')}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Stream:</span> <b style="color:#e2e8f0;">{row.get('Stream','N/A')}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Mode:</span> <b style="color:#e2e8f0;">{row.get('Mode','N/A')}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Total Fee:</span> <b style="color:#fbbf24;">₹{row.get('Total_Amount',0):,.0f}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Paid:</span> <b style="color:#34d399;">₹{row.get('Paid_amount',0):,.0f}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Outstanding:</span> <b style="color:#f87171;">₹{row.get('Outstanding_Amount',0):,.0f}</b></div>
            <div style="margin-bottom:8px;"><span style="color:#64748b;">Pay Method:</span> <b style="color:#e2e8f0;">{row.get('Payment_Method','N/A')}</b></div>
            <div style="margin-top:8px; background:rgba(0,0,0,0.3); border-radius:6px; height:8px; overflow:hidden;">
                <div style="width:{min(pr,100):.0f}%; background:linear-gradient(90deg,#6366f1,#34d399); height:100%; border-radius:6px;"></div>
            </div>
            <div style="font-size:11px; color:#64748b; margin-top:4px;">{pr:.1f}% paid</div>
        </div>
        """, unsafe_allow_html=True)

    with pc3:
        st.markdown(f"""
        <div class="candidate-card">
            <div style="font-size:13px; font-weight:700; color:#94a3b8; margin-bottom:12px; text-transform:uppercase; letter-spacing:1px;">Call History ({len(calls)} calls)</div>
        """, unsafe_allow_html=True)
        if not calls.empty:
            calls_disp = calls[['Call_Date','Call_Duration','Call_Remarks']].copy()
            calls_disp.columns = ['Date', 'Duration (min)', 'Remarks']
            calls_disp['Date'] = calls_disp['Date'].dt.strftime('%d %b %Y')
            st.dataframe(calls_disp.reset_index(drop=True), use_container_width=True, height=220)
        else:
            st.markdown("<p style='color:#475569;'>No call records found.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Show suggested churn reason for this candidate if available
        if 'Suggested_Churn_Reason' in row.index and pd.notna(row['Suggested_Churn_Reason']) and row['Suggested_Churn_Reason'] != '':
            st.markdown(f"<div style='margin-top:12px; padding:12px; border-radius:8px; background:rgba(248,113,113,0.06);'>"
                        f"<b>Suggested Churn Reason:</b> {row['Suggested_Churn_Reason']}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 3 — CALL LOG ANALYSIS
# ─────────────────────────────────────────────
def page_call_analysis(df, call_log_proc, executive_profile):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-phone"></i> Call Log Analysis</h1>
        <p>Understand call engagement patterns, executive performance, and sentiment signals.</p>
    </div>
    """, unsafe_allow_html=True)

    exec_map = executive_profile.set_index('Executive_ID')['Executive_Name'].to_dict()
    call_log_proc['Executive_Name'] = call_log_proc['Executive_ID'].map(exec_map)

    # ── KPIs ─────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    avg_dur = call_log_proc['Call_Duration'].mean()
    total_dur = call_log_proc['Call_Duration'].sum()
    avg_calls_per_cand = len(call_log_proc) / call_log_proc['Candidate_ID'].nunique()

    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-phone" style="color:#a78bfa"></i></div>
            <div class="kpi-title">Total Calls</div>
            <div class="kpi-value" style="color:#60a5fa">{len(call_log_proc)}</div>
            <div class="kpi-sub">Across all candidates</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-stopwatch"></i></div>
            <div class="kpi-title">Avg Duration</div>
            <div class="kpi-value" style="color:#a78bfa">{avg_dur:.1f} min</div>
            <div class="kpi-sub">Per call</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-rotate"></i></div>
            <div class="kpi-title">Avg Calls/Candidate</div>
            <div class="kpi-value" style="color:#34d399">{avg_calls_per_cand:.1f}</div>
            <div class="kpi-sub">Follow-up rate</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-chart-simple"></i></div>
            <div class="kpi-title">Total Talk Time</div>
            <div class="kpi-value" style="color:#fbbf24">{total_dur:.0f} min</div>
            <div class="kpi-sub">Combined duration</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header"><h2>Calls per Executive</h2></div>', unsafe_allow_html=True)
        exec_calls = call_log_proc.groupby('Executive_Name').agg(
            Calls=('Candidate_ID','count'),
            Avg_Duration=('Call_Duration','mean')
        ).reset_index().sort_values('Calls', ascending=True)
        fig = px.bar(exec_calls, x='Calls', y='Executive_Name', orientation='h',
                     color='Calls', color_continuous_scale=['#4c1d95','#6366f1','#06b6d4'],
                     text='Calls', hover_data={'Avg_Duration':':.2f'})
        fig.update_traces(textfont_size=11, textposition='outside')
        fig.update_layout(**theme(height=310, showlegend=False, coloraxis_showscale=False))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h2>Avg Call Duration: Active vs Churned</h2></div>', unsafe_allow_html=True)
        merged = df[['Candidate_ID','Churn']].merge(call_log_proc, on='Candidate_ID', how='right')
        merged['Status'] = merged['Churn'].map({0:'Active', 1:'Churned'}).fillna('Unknown')
        dur_by_status = merged.groupby('Status')['Call_Duration'].mean().reset_index()
        dur_by_status.columns = ['Status','Avg Duration (min)']
        fig2 = px.bar(dur_by_status, x='Status', y='Avg Duration (min)',
                      color='Status', color_discrete_map={'Active':COLOR_ACTIVE,'Churned':COLOR_CHURN},
                      text=dur_by_status['Avg Duration (min)'].round(2))
        fig2.update_traces(textfont_size=12, textposition='outside')
        fig2.update_layout(**theme(height=310, showlegend=False))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Sentiment Keywords ─────────────────────────
    st.markdown('<div class="section-header"><h2><i class="fa-solid fa-clipboard"></i> Call Remark Sentiment Keywords</h2></div>', unsafe_allow_html=True)

    col3, col4 = st.columns([1.5, 1])

    with col3:
        sentiment_data = {
            'Signal':    ['Interested / Confirmed', 'No Response / Unreachable',
                          'Payment / Fee / EMI', 'Technical / Syllabus / Mentor'],
            'Active':    [
                df[(df['Churn']==0) & (df['has_interest']==1)].shape[0] if 'has_interest' in df.columns else 0,
                df[(df['Churn']==0) & (df['has_no_response']==1)].shape[0] if 'has_no_response' in df.columns else 0,
                df[(df['Churn']==0) & (df['has_payment_discussion']==1)].shape[0] if 'has_payment_discussion' in df.columns else 0,
                df[(df['Churn']==0) & (df['has_technical_discussion']==1)].shape[0] if 'has_technical_discussion' in df.columns else 0,
            ],
            'Churned':   [
                df[(df['Churn']==1) & (df['has_interest']==1)].shape[0] if 'has_interest' in df.columns else 0,
                df[(df['Churn']==1) & (df['has_no_response']==1)].shape[0] if 'has_no_response' in df.columns else 0,
                df[(df['Churn']==1) & (df['has_payment_discussion']==1)].shape[0] if 'has_payment_discussion' in df.columns else 0,
                df[(df['Churn']==1) & (df['has_technical_discussion']==1)].shape[0] if 'has_technical_discussion' in df.columns else 0,
            ],
        }
        sent_df = pd.DataFrame(sentiment_data)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Active',  x=sent_df['Signal'], y=sent_df['Active'],  marker_color=COLOR_ACTIVE))
        fig3.add_trace(go.Bar(name='Churned', x=sent_df['Signal'], y=sent_df['Churned'], marker_color=COLOR_CHURN))
        fig3.update_layout(**theme(height=310, barmode='group',
                                   xaxis_tickangle=-20))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header"><h2>Remark Signal Summary</h2></div>', unsafe_allow_html=True)
        for sig, active_val, churn_val in zip(
            ['<i class="fa-solid fa-circle-check" style="color:#34d399"></i> Interested', '<i class="fa-solid fa-circle-xmark" style="color:#f87171"></i> No Response', '<i class="fa-solid fa-sack-dollar" style="color:#fbbf24"></i> Payment Talk', '<i class="fa-solid fa-lightbulb" style="color:#a78bfa"></i> Technical Talk'],
            sentiment_data['Active'], sentiment_data['Churned']
        ):
            st.markdown(f"""
            <div style="background:rgba(30,30,60,0.5); border:1px solid rgba(255,255,255,0.15);
                        border-radius:10px; padding:12px 16px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#e2e8f0; font-size:13px;">{sig}</span>
                <span>
                    <span style="color:#34d399; font-weight:700;">{active_val} active</span>
                    &nbsp;·&nbsp;
                    <span style="color:#f87171; font-weight:700;">{churn_val} churned</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ── Call Timeline ─────────────────────────────
    st.markdown('<div class="section-header"><h2><i class="fa-solid fa-calendar-days"></i> Call Activity Timeline</h2></div>', unsafe_allow_html=True)
    timeline = call_log_proc.groupby(call_log_proc['Call_Date'].dt.date).size().reset_index()
    timeline.columns = ['Date', 'Calls']
    timeline['Date'] = pd.to_datetime(timeline['Date'])
    fig4 = go.Figure(go.Scatter(
        x=timeline['Date'], y=timeline['Calls'],
        mode='lines+markers',
        line=dict(color='#6366f1', width=2.5),
        marker=dict(size=7, color='#a78bfa'),
        fill='tozeroy', fillcolor='rgba(255,255,255,0.1)',
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Calls: %{y}<extra></extra>'
    ))
    fig4.update_layout(**theme(
        height=260,
        xaxis=dict(title='Date', gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False),
        yaxis=dict(title='Number of Calls', gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False),
    ))
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 4 — PAYMENT ANALYSIS
# ─────────────────────────────────────────────
def page_payment_analysis(df):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-money-bill-wave"></i> Payment Analysis</h1>
        <p>Fee collection status, payment methods, and financial risk by churn segment.</p>
    </div>
    """, unsafe_allow_html=True)

    total_revenue    = df['Total_Amount'].sum()
    collected        = df['Paid_amount'].sum()
    outstanding      = df['Outstanding_Amount'].sum()
    collection_rate  = collected / total_revenue * 100 if total_revenue > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-money-bill"></i></div>
            <div class="kpi-title">Total Expected</div>
            <div class="kpi-value" style="color:#60a5fa; font-size:26px;">₹{total_revenue/1e5:.2f}L</div>
            <div class="kpi-sub">Course fees total</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-check-circle" style="color:#34d399"></i></div>
            <div class="kpi-title">Collected</div>
            <div class="kpi-value" style="color:#34d399; font-size:26px;">₹{collected/1e5:.2f}L</div>
            <div class="kpi-sub">{collection_rate:.1f}% collected</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-triangle-exclamation" style="color:#f87171"></i></div>
            <div class="kpi-title">Outstanding</div>
            <div class="kpi-value" style="color:#f87171; font-size:26px;">₹{outstanding/1e5:.2f}L</div>
            <div class="kpi-sub">Pending recovery</div></div>""", unsafe_allow_html=True)
    with k4:
        churn_outstanding = df[df['Churn']==1]['Outstanding_Amount'].sum()
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-bell"></i></div>
            <div class="kpi-title">At-Risk Amount</div>
            <div class="kpi-value" style="color:#fbbf24; font-size:26px;">₹{churn_outstanding/1e5:.2f}L</div>
            <div class="kpi-sub">From churned candidates</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header"><h2>Payment Ratio Distribution</h2></div>', unsafe_allow_html=True)
        fig = go.Figure()
        for churn_val, label, color in [(0,'Active',COLOR_ACTIVE),(1,'Churned',COLOR_CHURN)]:
            sub = df[df['Churn']==churn_val]['Payment_Ratio'] * 100
            fig.add_trace(go.Histogram(x=sub, name=label, nbinsx=12,
                                       marker_color=color, opacity=0.75,
                                       hovertemplate=f'<b>{label}</b><br>Pay%: %{{x:.0f}}%<br>Count: %{{y}}<extra></extra>'))
        fig.update_layout(**theme(height=300, barmode='overlay',
                                  xaxis_title='Payment % of Total Fee',
                                  yaxis_title='Number of Candidates'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h2>Payment Method Breakdown</h2></div>', unsafe_allow_html=True)
        pm = df.groupby(['Payment_Method','Churn']).size().reset_index(name='Count')
        pm['Status'] = pm['Churn'].map({0:'Active',1:'Churned'})
        pm['Payment_Method'].fillna('Unknown', inplace=True)
        fig2 = px.bar(pm, x='Payment_Method', y='Count', color='Status',
                      color_discrete_map={'Active':COLOR_ACTIVE,'Churned':COLOR_CHURN},
                      barmode='group', text='Count')
        fig2.update_traces(textfont_size=11, textposition='outside')
        fig2.update_layout(**theme(height=300, showlegend=True))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Fee vs Paid scatter ───────────────────────
    st.markdown('<div class="section-header"><h2>Total Fee vs Amount Paid (per Candidate)</h2></div>', unsafe_allow_html=True)
    scatter_df = df.copy()
    scatter_df['Status'] = scatter_df['Churn'].map({0:'Active',1:'Churned'})
    fig3 = px.scatter(scatter_df, x='Total_Amount', y='Paid_amount',
                      color='Status', size='Total_Calls',
                      color_discrete_map={'Active':COLOR_ACTIVE,'Churned':COLOR_CHURN},
                      hover_data=['Candidate_Name','Course','Payment_Method'],
                      labels={'Total_Amount':'Total Course Fee (₹)','Paid_amount':'Amount Paid (₹)'},
                      size_max=20)
    # diagonal line
    max_val = df['Total_Amount'].max()
    fig3.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                              mode='lines', name='100% Paid',
                              line=dict(dash='dash', color='rgba(255,255,255,0.2)', width=1)))
    fig3.update_layout(**theme(height=360))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Course-wise revenue ────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header"><h2>Revenue by Course</h2></div>', unsafe_allow_html=True)
        rev = df.groupby('Course').agg(
            Collected=('Paid_amount','sum'),
            Outstanding=('Outstanding_Amount','sum')
        ).reset_index()
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name='Collected',    x=rev['Course'], y=rev['Collected'],    marker_color='#34d399'))
        fig4.add_trace(go.Bar(name='Outstanding',  x=rev['Course'], y=rev['Outstanding'],  marker_color='#f87171'))
        fig4.update_layout(**theme(height=310, barmode='stack',
                                   yaxis_title='Amount (₹)', xaxis_tickangle=-20))
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header"><h2>Avg Payment Ratio by Course</h2></div>', unsafe_allow_html=True)
        avg_pr = df.groupby('Course')['Payment_Ratio'].mean().reset_index()
        avg_pr['Payment_%'] = (avg_pr['Payment_Ratio'] * 100).round(1)
        avg_pr = avg_pr.sort_values('Payment_%')
        fig5 = px.bar(avg_pr, x='Payment_%', y='Course', orientation='h',
                      color='Payment_%', color_continuous_scale=['#ef4444','#f59e0b','#10b981'],
                      text=avg_pr['Payment_%'].astype(str)+'%')
        fig5.update_traces(textfont_size=11, textposition='outside')
        fig5.update_layout(**theme(height=310, showlegend=False, coloraxis_showscale=False,
                                   xaxis_title='Average Payment %'))
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 5 — LIVE PREDICTOR
# ─────────────────────────────────────────────
def page_live_predictor(df, model_data, churn_full=None):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-robot"></i> Live Churn Predictor</h1>
        <p>Fill in candidate details to get an instant AI-powered churn risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_data is None:
        st.error("Could not load churn_prediction_model.pkl. Please ensure the model file is present in the project directory.")
        return

    model           = model_data['model']
    scaler          = model_data['scaler']
    feature_columns = model_data['feature_columns']
    label_encoders  = model_data['label_encoders']
    categorical_feat= model_data['categorical_features']
    numerical_feat  = model_data['numerical_features']
    balance_method  = model_data.get('balance_method', 'none')

    st.markdown('<div class="section-header"><h2>Candidate Details</h2></div>', unsafe_allow_html=True)
    st.markdown(f"**Model:** {model_data.get('model_display_name', 'Unknown')}  •  **Balancing:** {format_balance_method(balance_method)}")

    col1, col2, col3 = st.columns(3)
    with col1:
        source    = st.selectbox("Source",     sorted(df['Source'].unique()), key="p_source")
        education = st.selectbox("Education",  sorted(df['Education'].unique()), key="p_edu")
        background= st.selectbox("Background", sorted(df['Background'].unique()), key="p_bg")
        role      = st.selectbox("Role",       sorted(df['Role'].unique()), key="p_role")

    with col2:
        stream    = st.selectbox("Stream",     sorted(df['Stream'].unique()), key="p_stream")
        course    = st.selectbox("Course",     sorted(df['Course'].unique()), key="p_course")
        mode      = st.selectbox("Mode",       sorted(df['Mode'].unique()), key="p_mode")
        pay_method= st.selectbox("Payment Method", sorted(df['Payment_Method'].dropna().unique()), key="p_pm")

    with col3:
        current_status = st.selectbox("Current Status", sorted(df['Current_status'].unique()), key="p_cs")
        induction_session = st.selectbox(
            "Induction Session",
            sorted(df['Induction_Session'].dropna().unique()) if 'Induction_Session' in df.columns else ['Attended', 'NotAttended'],
            key="p_is"
        )
        feedback = st.selectbox(
            "Feedback",
            sorted(df['Feedback'].dropna().unique()) if 'Feedback' in df.columns else ['Positive', 'Neutral', 'Negative'],
            key="p_fb"
        )
        experience     = st.number_input("Experience (years)", 0, 30, 3, key="p_exp")
        career_gap     = st.number_input("Career Gap (years)", 0, 10, 0, key="p_gap")
        total_amount   = st.number_input("Total Course Fee (₹)", 10000, 200000, 60000, step=5000, key="p_ta")
        paid_amount    = st.number_input("Amount Paid (₹)", 0, 200000, 0, step=5000, key="p_pa")

    st.markdown('<div class="section-header"><h2>Call History Inputs</h2></div>', unsafe_allow_html=True)

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        total_calls = st.number_input("Total Calls", 0, 5, 1, key="p_tc")
        if total_calls == 0:
            st.number_input("Unique Executives", value=0, disabled=True, key="p_ue_disabled")
            unique_execs = 0
        else:
            unique_execs = st.number_input("Unique Executives", 1, 3, 1, key="p_ue")

    with cc2:
        if total_calls == 0:
            st.number_input("Total Call Duration (min)", value=0.0, disabled=True, key="p_tcd_disabled")
            total_call_dur = 0.0
        else:
            total_call_dur = st.number_input("Total Call Duration (min)", 0.0, 30.0, 8.0, step=0.5, key="p_tcd")

        if total_calls <= 1:
            st.number_input("Avg Call Duration (min)", value=float(total_call_dur), disabled=True, key="p_acd_disabled")
            avg_call_dur = total_call_dur
        else:
            avg_call_dur = st.number_input("Avg Call Duration (min)", 0.0, 30.0, 4.0, step=0.5, key="p_acd")

    with cc3:
        if total_calls <= 1:
            st.number_input("Max Call Duration (min)", value=float(total_call_dur), disabled=True, key="p_mxcd_disabled")
            max_call_dur = total_call_dur
            st.number_input("Min Call Duration (min)", value=float(total_call_dur), disabled=True, key="p_mncd_disabled")
            min_call_dur = total_call_dur
        else:
            max_call_dur = st.number_input("Max Call Duration (min)", 0.0, 30.0, 6.0, step=0.5, key="p_mxcd")
            min_call_dur = st.number_input("Min Call Duration (min)", 0.0, 30.0, 2.0, step=0.5, key="p_mncd")

    with cc4:
        if total_calls == 0:
            st.number_input("Call Frequency (per month)", value=0.0, disabled=True, key="p_cf_disabled")
            call_freq = 0.0
        else:
            call_freq = st.number_input("Call Frequency (per month)", 0.0, 30.0, 0.18, step=0.01, key="p_cf")

        exec_exp = st.number_input("Avg Executive Experience (yrs)", 0.0, 10.0, 5.0, step=0.5, key="p_ee")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        if total_calls == 0:
            has_interest = st.checkbox("Showed Interest in Calls", value=False, disabled=True, key="p_hi_disabled")
        else:
            has_interest = st.checkbox("Showed Interest in Calls", value=True, key="p_hi")
    with sc2:
        if total_calls == 0:
            has_no_resp = st.checkbox("No Response / Unreachable", value=False, disabled=True, key="p_hnr_disabled")
        else:
            has_no_resp = st.checkbox("No Response / Unreachable", value=False, key="p_hnr")
    with sc3:
        if total_calls == 0:
            has_payment = st.checkbox("Payment Discussion in Calls", value=False, disabled=True, key="p_hpd_disabled")
        else:
            has_payment = st.checkbox("Payment Discussion in Calls", value=False, key="p_hpd")
    with sc4:
        if total_calls == 0:
            has_technical = st.checkbox("Technical Discussion", value=False, disabled=True, key="p_htd_disabled")
        else:
            has_technical = st.checkbox("Technical Discussion", value=True, key="p_htd")

    # Free-text call remarks and optional transcript for live inference
    call_remarks = st.text_area("Call Remarks (optional)", value="", max_chars=1000, placeholder="Enter recent call remarks or notes...", key="p_remarks")
    call_transcript = st.text_area("Call Transcript (optional)", value="", max_chars=2000, placeholder="Paste full call transcript to improve AI churn reason extraction.", key="p_transcript")

    exec_team = st.selectbox("Executive Team", sorted(df['Executive_Team'].dropna().unique()) if 'Executive_Team' in df.columns else ['Team A','Team B','Team C','Team D'], key="p_et")

    days_since_payment = st.number_input("Days Since Last Payment", 0, 1000, 60, key="p_dsp")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Churn Risk", use_container_width=True, type="primary"):
        payment_ratio = paid_amount / total_amount if total_amount > 0 else 0.0

        raw_input = {
            'Source':              source,
            'Education':           education,
            'Background':          background,
            'Role':                role,
            'Current_status':      current_status,
            'Stream':              stream,
            'Course':              course,
            'Mode':                mode,
            'Payment_Method':      pay_method,
            'Executive_Team':      exec_team,
            'Induction_Session':   induction_session,
            'Feedback':            feedback,
            'Experience':          experience,
            'Career_gap':          career_gap,
            'Total_Amount':        total_amount,
            'Paid_amount':         paid_amount,
            'Payment_Ratio':       payment_ratio,
            'Zero_Payment':        int(paid_amount == 0),
            'Negative_Feedback':   int(str(feedback).strip().lower() == 'negative'),
            'High_Risk_Indicator': int((paid_amount == 0) and (str(feedback).strip().lower() == 'negative')),
            'Days_Since_Induction': 30,
            'Days_Since_Payment':  days_since_payment,
            'Total_Calls':         total_calls,
            'Unique_Executives':   unique_execs,
            'Total_Call_Duration': total_call_dur,
            'Avg_Call_Duration':   avg_call_dur,
            'Max_Call_Duration':   max_call_dur,
            'Min_Call_Duration':   min_call_dur,
            'Call_Frequency':      call_freq,
            'Executive_Experience': exec_exp,
            'has_interest':        int(has_interest),
            'has_no_response':     int(has_no_resp),
            'has_payment_discussion': int(has_payment),
            'has_technical_discussion': int(has_technical),
        }

        input_df = pd.DataFrame([raw_input])

        # Add any missing model features with safe defaults
        for col in feature_columns:
            if col not in input_df.columns:
                if col in numerical_feat:
                    input_df[col] = 0
                else:
                    input_df[col] = 'No Contact' if col == 'Executive_Team' else 'Unknown'

        # Encode categoricals using saved label encoders
        for col in categorical_feat:
            if col in input_df.columns and col in label_encoders:
                le = label_encoders[col]
                val = str(input_df[col].iloc[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = 0  # unseen category → default 0

        # Build feature vector
        X = input_df[feature_columns].copy()
        for col in numerical_feat:
            if col not in X.columns:
                X[col] = 0

        X[numerical_feat] = scaler.transform(X[numerical_feat].values.reshape(1, -1))

        prob = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

        # ── Result Display ─────────────────────────
        r1, r2, r3 = st.columns([1.2, 1.2, 1.6])

        with r1:
            if pred == 1:
                st.markdown(f"""
                <div class="prediction-box-churn">
                    <div class="pred-label" style="color:#f87171;"><i class="fa-solid fa-circle-xmark"></i></div>
                    <div style="font-size:24px; font-weight:800; color:#f87171; margin-bottom:8px;">HIGH CHURN RISK</div>
                    <div class="pred-sub">This candidate is likely to NOT join training</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-safe">
                    <div class="pred-label" style="color:#34d399;"><i class="fa-solid fa-circle-check"></i></div>
                    <div style="font-size:24px; font-weight:800; color:#34d399; margin-bottom:8px;">LOW CHURN RISK</div>
                    <div class="pred-sub">This candidate is likely to join training</div>
                </div>""", unsafe_allow_html=True)

        with r2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                title={"text": "Churn Probability", "font": {"color": "#94a3b8", "size": 14}},
                number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar":  {"color": "#f87171" if prob > 0.5 else "#34d399", "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 30],   "color": "rgba(52,211,153,0.15)"},
                        {"range": [30, 60],  "color": "rgba(251,191,36,0.15)"},
                        {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {"value": 50, "line": {"color": "#fff", "width": 2}, "thickness": 0.8}
                }
            ))
            gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#94a3b8'),
                                 margin=dict(l=20, r=20, t=40, b=20), height=250)
            st.plotly_chart(gauge, use_container_width=True)

        with r3:
            risk_level = "High" if prob > 0.6 else ("Medium" if prob > 0.35 else "Low")
            st.markdown(f"""
            <div class="candidate-card" style="margin-top:0;">
                <div style="font-size:13px; font-weight:700; color:#94a3b8; margin-bottom:14px; text-transform:uppercase; letter-spacing:1px;">Risk Assessment</div>
                <div style="margin-bottom:10px; display:flex; justify-content:space-between;">
                    <span style="color:#64748b;">Churn Probability</span>
                    <b style="color:#e2e8f0;">{prob*100:.1f}%</b>
                </div>
                <div style="margin-bottom:10px; display:flex; justify-content:space-between;">
                    <span style="color:#64748b;">Risk Level</span>
                    <b style="color:#e2e8f0;">{risk_level}</b>
                </div>
                <div style="margin-bottom:10px; display:flex; justify-content:space-between;">
                    <span style="color:#64748b;">Payment Ratio</span>
                    <b style="color:#e2e8f0;">{payment_ratio*100:.1f}%</b>
                </div>
                <div style="margin-bottom:10px; display:flex; justify-content:space-between;">
                    <span style="color:#64748b;">Call Engagement</span>
                    <b style="color:#e2e8f0;">{total_calls} calls / {total_call_dur:.1f} min total</b>
                </div>
                <hr style="border-color:rgba(255,255,255,0.2); margin:12px 0;">
                <div style="font-size:12px; color:#64748b;">
                    {'<i class="fa-solid fa-triangle-exclamation" style="color:#fbbf24"></i> <b style="color:#fbbf24;">Action Required:</b> Schedule immediate follow-up call.' if pred==1 else '<i class="fa-solid fa-circle-check" style="color:#34d399"></i> <b style="color:#34d399;">On Track:</b> Continue regular follow-up.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        suggested_reason, ai_recommendation, extraction_method = extract_reason_and_recommendation(
            raw_input,
            call_remarks,
            feedback,
            call_transcript
        )

        st.markdown(f"<div style='margin-top:12px; padding:12px; border-radius:8px; background:rgba(255,255,255,0.06);'>"
                    f"<b><small style='color:#f8fafc;'>Suggested Churn Reason:</b> {suggested_reason}<br>"
                    f"<small style='color:#94a3b8;'>Extraction method: {extraction_method}</small>"
                    f"</div>", unsafe_allow_html=True)

        if ai_recommendation:
            st.markdown(f"<div style='margin-top:12px; padding:14px; border-radius:10px; background:rgba(52,211,153,0.08);'>"
                        f"<b><small style='color:#f8fafc;'>AI-Generated Recommendation:</b> {ai_recommendation}" 
                        f"</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 6 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
def page_model_performance(df, model_data):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-chart-line"></i> Model Performance</h1>
        <p>Evaluation metrics, feature importance, and model comparison for the churn prediction model.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_data is None:
        st.error("Model file not found.")
        return

    model           = model_data['model']
    feature_columns = model_data['feature_columns']
    balance_method  = model_data.get('balance_method', 'None')
    balance_label   = format_balance_method(balance_method)
    balance_note    = balance_method_description(balance_method)
    model_name      = model_data.get('model_display_name') or model_data.get('model_name') or model.__class__.__name__
    friendly_model  = {
        'RandomForestClassifier': 'Random Forest',
        'GradientBoostingClassifier': 'Gradient Boosting',
        'XGBClassifier': 'XGBoost',
        'LogisticRegression': 'Logistic Regression',
        'Random Forest (Regularized)': 'Random Forest (Regularized)',
        'Gradient Boosting (Regularized)': 'Gradient Boosting (Regularized)',
        'XGBoost (Regularized)': 'XGBoost (Regularized)'
    }.get(model_name, model_name)

    balance_select_reason = {
        'none': 'No balancing method was selected because the original training distribution achieved the best validation performance.',
        'class_weight': 'Class weights were chosen because they improved minority churn detection while keeping the full training set intact.',
        'oversample': 'Random oversampling was selected because it produced the best validation F1 for the minority churn class.',
        'undersample': 'Random undersampling was selected because it improved validation performance by balancing class representation.',
        'smote': 'SMOTE was selected because synthetic minority samples improved validation F1 and class generalization.'
    }.get(str(balance_method).lower(), 'Selected based on validation performance during class imbalance evaluation.')

    selected_model = friendly_model

    # ── Model Info ────────────────────────────────
    st.markdown('<div class="section-header"><h2>Model Information</h2></div>', unsafe_allow_html=True)
    i1, i2, i3, i4, i5 = st.columns(5)
    with i1:
        st.markdown(f"""<div class=\"kpi-card\"><div class=\"kpi-icon\"><i class="fa-solid fa-robot"></i></div>
            <div class=\"kpi-title\">Algorithm</div>
            <div class=\"kpi-value\" style=\"color:#a78bfa; font-size:16px; margin-top:8px;\">{friendly_model}</div>
            <div class=\"kpi-sub\">Deployed model</div></div>""", unsafe_allow_html=True)
    with i2:
        estimator_value = getattr(model,'n_estimators', None) or getattr(model,'n_estimators_', 'N/A')
        st.markdown(f"""<div class=\"kpi-card\"><div class=\"kpi-icon\"><i class="fa-solid fa-tree"></i></div>
            <div class=\"kpi-title\">Estimators</div>
            <div class=\"kpi-value\" style=\"color:#60a5fa;\">{estimator_value}</div>
            <div class=\"kpi-sub\">Decision trees</div></div>""", unsafe_allow_html=True)
    with i3:
        max_depth_value = getattr(model,'max_depth', 'N/A')
        st.markdown(f"""<div class=\"kpi-card\"><div class=\"kpi-icon\"><i class="fa-solid fa-ruler-combined"></i></div>
            <div class=\"kpi-title\">Max Depth</div>
            <div class=\"kpi-value\" style=\"color:#34d399;\">{max_depth_value}</div>
            <div class=\"kpi-sub\">Tree depth limit</div></div>""", unsafe_allow_html=True)
    with i4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-hashtag"></i></div>
            <div class="kpi-title">Features Used</div>
            <div class="kpi-value" style="color:#fbbf24;">{len(feature_columns)}</div>
            <div class="kpi-sub">Input dimensions</div></div>""", unsafe_allow_html=True)
    with i5:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-scale-balanced"></i></div>
            <div class="kpi-title">Balancing</div>
            <div class="kpi-value" style="color:#fbbf24; font-size:16px; margin-top:8px;">{balance_label}</div>
            <div class="kpi-sub">Imbalance handling</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>Balancing Technique</h2></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="candidate-card">
        <div style="display:flex; justify-content:space-between; gap:16px; align-items:center; flex-wrap:wrap;">
            <div>
                <div style="font-size:13px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Selected Method</div>
                <div style="font-size:24px; font-weight:800; color:#fbbf24; margin-top:4px;">{balance_label}</div>
            </div>
            <div style="max-width:620px; color:#94a3b8; font-size:13px; line-height:1.5;">{balance_note}</div>
        </div>
        <div style="margin-top:14px; color:#94a3b8; font-size:13px; line-height:1.6;">
            <strong>Why this method?</strong> {balance_select_reason}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Importance ─────────────────────────
    st.markdown('<div class="section-header"><h2>Feature Importance (Top 15)</h2></div>', unsafe_allow_html=True)

    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'Feature':    feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(fi.sort_values('Importance'), x='Importance', y='Feature',
                         orientation='h',
                         color='Importance', color_continuous_scale=['#4c1d95','#6366f1','#06b6d4'],
                         text=fi.sort_values('Importance')['Importance'].round(3))
            fig.update_traces(textfont_size=10, textposition='outside')
            fig.update_layout(**theme(height=420, showlegend=False,
                                      coloraxis_showscale=False, xaxis_title='Importance Score'))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<div style='padding-top:8px;'>", unsafe_allow_html=True)
            for _, row in fi.iterrows():
                pct = row['Importance'] / fi['Importance'].max() * 100
                st.markdown(f"""
                <div style="margin-bottom:6px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                        <span style="font-size:11px; color:#94a3b8;">{row['Feature']}</span>
                        <span style="font-size:11px; color:#a78bfa;">{row['Importance']:.3f}</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.1); border-radius:4px; height:5px;">
                        <div style="width:{pct:.0f}%; background:linear-gradient(90deg,#6366f1,#8b5cf6); height:100%; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Churn Reason Analysis ─────────────────────
    st.markdown('<div class="section-header"><h2><i class="fa-solid fa-clipboard-list"></i> Why Are Candidates Churning? — Reason Analysis</h2></div>', unsafe_allow_html=True)

    churned_df = df[df['Churn'] == 1].copy()
    active_df  = df[df['Churn'] == 0].copy()

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown("""<div class="candidate-card">
                       <div style="font-size:13px; font-weight:700; color:#f87171; margin-bottom:12px;"><i class="fa-solid fa-circle-xmark"></i> No interest</div>""", unsafe_allow_html=True)
        no_int = churned_df.get('has_no_interest', pd.Series([0]*len(churned_df))).sum() if 'has_no_interest' in churned_df.columns else 0
        pct = no_int / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
            <div style="font-size:32px; font-weight:800; color:#fbbf24;">{int(no_int)}</div>
            <div style="color:#64748b; font-size:13px;">churned had no interest ({pct:.0f}%)</div>
            <div style="margin-top:10px; font-size:12px; color:#475569;">Churned candidates made initial payment - but no longer interested, they may have attended or not attended the induction session.</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("""<div class="candidate-card">
                       <div style="font-size:13px; font-weight:700; color:#fbbf24; margin-bottom:12px;"><i class="fa-solid fa-phone-slash"></i> No Response Pattern</div>""", unsafe_allow_html=True)
        no_resp = churned_df.get('has_no_response', pd.Series([0]*len(churned_df))).sum() if 'has_no_response' in churned_df.columns else 0
        pct2 = no_resp / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
            <div style="font-size:32px; font-weight:800; color:#fbbf24;">{int(no_resp)}</div>
            <div style="color:#64748b; font-size:13px;">churned had no-response calls ({pct2:.0f}%)</div>
            <div style="margin-top:10px; font-size:12px; color:#475569;">Unreachable candidates rarely convert — early escalation is critical.</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown("""<div class="candidate-card">
                       <div style="font-size:13px; font-weight:700; color:#10b981; margin-bottom:12px;"><i class="fa-solid fa-money-bill-wave"></i> Financial Issue</div>""", unsafe_allow_html=True)
        fin_issue = churned_df.get('has_payment_discussion', pd.Series([0]*len(churned_df))).sum() if 'has_payment_discussion' in churned_df.columns else 0
        pct_fin = fin_issue / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
        <div style="font-size:32px; font-weight:800; color:#f87171;">{int(fin_issue)}</div>
        <div style="color:#64748b; font-size:13px;">Churned with payment discussion ({pct_fin:.0f}%)</div>
        <div style="margin-top:10px; font-size:12px; color:#475569;">Candidates who had financial concerns or payment-related discussions before churning.</div>
        </div>""", unsafe_allow_html=True)

    with r4:
        st.markdown("""<div class="candidate-card">
                       <div style="font-size:13px; font-weight:700; color:#3b82f6; margin-bottom:12px;"><i class="fa-solid fa-graduation-cap"></i> Joined Another Institution</div>""", unsafe_allow_html=True)
        joined_other = churned_df.get('joined_another', pd.Series([0]*len(churned_df))).sum() if 'joined_another' in churned_df.columns else 0
        pct_joined = joined_other / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
        <div style="font-size:32px; font-weight:800; color:#f87171;">{int(joined_other)}</div>
        <div style="color:#64748b; font-size:13px;">Churned to join another institute ({pct_joined:.0f}%)</div>
        <div style="margin-top:10px; font-size:12px; color:#475569;">Candidates who chose to enroll at a competitor institution instead.</div>
        </div>""", unsafe_allow_html=True)        

    # ── Model Comparison Table (from model.py's logic) ──
    st.markdown('<div class="section-header"><h2><i class="fa-solid fa-trophy"></i> Algorithm Comparison (Reference)</h2></div>', unsafe_allow_html=True)

    model_names = ['Random Forest (Regularized)', 'Gradient Boosting (Regularized)',
                   'XGBoost (Regularized)', 'Random Forest', 'Gradient Boosting',
                   'XGBoost', 'Logistic Regression', 'AdaBoost', 'Decision Tree', 'SVM', 'KNN', 'Naive Bayes']
    model_strengths = ['Balanced accuracy, low overfit', 'High F1, slight overfit',
                       'High accuracy', 'Good recall', 'High precision',
                       'Fast', 'Interpretable', 'Ensemble', 'Fast', 'High precision', 'Simple', 'Probabilistic']

    notes = []
    for name in model_names:
        if name == selected_model:
            notes.append('Selected')
        elif 'Regularized' in name:
            notes.append('Tuned')
        else:
            notes.append('Baseline')

    comparison_data = {
        'Model': model_names,
        'Note': notes,
        'Strength': model_strengths,
    }
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Model":    st.column_config.TextColumn("Algorithm"),
                     "Note":     st.column_config.TextColumn("Status"),
                     "Strength": st.column_config.TextColumn("Characteristic"),
                 })


def page_profile():
    st.markdown('<div class="page-header"><h1><i class="fa-solid fa-user-circle"></i> Executive Profile</h1><p>Manage your account settings and personal info.</p></div>', unsafe_allow_html=True)
    
    try:
        user_res = supabase.auth.get_user()
        if not user_res or not user_res.user:
            st.error("Could not fetch user session.")
            return
        u = user_res.user
    except Exception as e:
        st.error(f"Error fetching profile: {e}")
        return

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2); border-radius:10px; padding:20px; margin-bottom:20px;">
        <h3 style="margin-top:0;">Account Information</h3>
        <p style="margin:5px 0;"><strong>Email ID:</strong> {u.email}</p>
        <p style="margin:5px 0;"><strong>User ID:</strong> {u.id}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header"><h2>Profile Details</h2></div>', unsafe_allow_html=True)
        current_name = u.user_metadata.get("full_name", "") if u.user_metadata else ""
        new_name = st.text_input("Display Name", value=current_name)
        if st.button("Save Profile", type="primary"):
            try:
                supabase.auth.update_user({"data": {"full_name": new_name}})
                st.success("Profile updated!")
            except Exception as e:
                st.error(f"Failed to update profile: {e}")

    with c2:
        st.markdown('<div class="section-header"><h2>Update Password</h2></div>', unsafe_allow_html=True)
        new_pass = st.text_input("New Password", type="password")
        if st.button("Update Password"):
            if new_pass:
                try:
                    supabase.auth.update_user({"password": new_pass})
                    st.success("Password updated!")
                except Exception as e:
                    st.error(f"Failed to update password: {e}")
            else:
                st.warning("Please enter a new password.")


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    if not st.session_state.get("logged_in", False):
        page_auth()
        return

    # Top header bar
    st.markdown("""
        <style>
            .profile-btn button {
                border-radius: 50px !important;
                border: 1px solid rgba(255,255,255,0.4) !important;
                background: rgba(30,41,59,0.5) !important;
                color: #e2e8f0 !important;
                font-weight: 600 !important;
            }
            .profile-btn button:hover {
                background: rgba(255,255,255,0.2) !important;
                border-color: #8b5cf6 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    c_left, c_right = st.columns([10, 1])
    with c_right:
        st.markdown('<div class="profile-btn">', unsafe_allow_html=True)
        if st.session_state.get("show_profile", False):
            if st.button("Back", icon=":material/arrow_back:", use_container_width=True):
                st.session_state.show_profile = False
                st.rerun()
        else:
            if st.button("Profile", icon=":material/person:", use_container_width=True):
                st.session_state.show_profile = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get("show_profile", False):
        sidebar()
        page_profile()
        return

    page = sidebar()

    # Load data
    with st.spinner("Loading data..."):
        try:
            candidate_profile, call_log, executive_profile = load_data()
            df, call_log_proc = preprocess(candidate_profile, call_log, executive_profile)
            churn_full, churn_short = load_churn_reasons()
        except FileNotFoundError as e:
            st.error(f"Could not load data files: {e}\n\nPlease ensure the CSV files are in the same directory as dashboard.py.")
            st.stop()

    model_path = os.path.join(OUTPUT_DIR, "churn_prediction_model.pkl")
    model_modified_time = os.path.getmtime(model_path) if os.path.exists(model_path) else None
    model_data = load_model(model_path, model_modified_time)

    if   "Overview"            in page: page_overview(df, call_log_proc, churn_full)
    elif "Candidate Explorer"  in page: page_candidate_explorer(df, call_log_proc, executive_profile, churn_full)
    elif "Call Log"            in page: page_call_analysis(df, call_log_proc, executive_profile)
    elif "Payment"             in page: page_payment_analysis(df)
    elif "Predictor"           in page: page_live_predictor(df, model_data)
    elif "Model Performance"   in page: page_model_performance(df, model_data)


if __name__ == "__main__":
    main()
