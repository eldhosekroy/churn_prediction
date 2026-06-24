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
import markdown
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

# Load company logo SVG
try:
    with open("RP2_full.svg", "r", encoding="utf-8") as f:
        svg_logo = f.read()
except Exception:
    svg_logo = None

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

def extract_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text=None, preferred_ai="Auto"):
    errors = []
    
    if preferred_ai in ["Auto (Fallback)", "Gemini"]:
        api_response = call_gemini_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text)
        if api_response and api_response.get('status') == 'AI (Gemini)' and api_response.get('reason'):
            return api_response['reason'], api_response.get('recommendation', ''), api_response['status']
        if api_response and 'error' in api_response:
            errors.append(f"Gemini: {api_response['error']}")
            
    if preferred_ai in ["Auto (Fallback)", "Groq"]:
        groq_response = call_groq_reason_and_recommendation(candidate_info, remarks_text, feedback_text, transcript_text)
        if groq_response and groq_response.get('status') == 'AI (Groq)' and groq_response.get('reason'):
            return groq_response['reason'], groq_response.get('recommendation', ''), groq_response['status']
        if groq_response and 'error' in groq_response:
            errors.append(f"Groq: {groq_response['error']}")
            
    if preferred_ai in ["Auto (Fallback)", "Hugging Face"]:
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
    page_icon="RP2.png",
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

    /* Equal-height profile cards — stretch all Streamlit column children to the tallest column */
    div[data-testid="stHorizontalBlock"]:has(.candidate-card) {
        align-items: stretch;
    }
    div[data-testid="stHorizontalBlock"]:has(.candidate-card) > div[data-testid="stColumn"] {
        display: flex;
        flex-direction: column;
    }
    div[data-testid="stHorizontalBlock"]:has(.candidate-card) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
        display: flex;
        flex-direction: column;
        flex: 1;
    }
    div[data-testid="stHorizontalBlock"]:has(.candidate-card) .candidate-card {
        flex: 1;
    }

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
OUTPUT_DIR = "./output/"

@st.cache_data
def load_data():
    # Load the processed dataset with inferred churn and reasons
    df_path = os.path.join(OUTPUT_DIR, "enrolled_processed.csv")
    notes_path = os.path.join(OUTPUT_DIR, "notes_processed.csv")
    
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame()
        
    if os.path.exists(notes_path):
        notes = pd.read_csv(notes_path)
    else:
        notes = pd.DataFrame()
        
    return df, notes

@st.cache_data
def load_churn_reasons():
    # Deprecated/Not needed anymore, returning same df for compatibility
    return None, None

@st.cache_data
def preprocess(df, notes):
    # Data is already preprocessed by churn_data.py.
    # We just ensure certain columns exist to avoid KeyError in UI
    
    if not df.empty:
        df['Contact Name'] = df['Contact Name'].fillna('Unknown')
        df['Course'] = df['Course'].fillna('Unknown')
        df['Source of lead'] = df['Source of lead'].fillna('Unknown')
        df['Mode of Program Joined'] = df['Mode of Program Joined'].fillna('Unknown')
        df['background'] = df['background'].fillna('Unknown')
        df['role'] = df['role'].fillna('Unknown')
        df['Invoice'] = df['Invoice'].fillna('No')
        
        # Payment mapping (if needed for legacy UI logic, though we will replace UI)
        #df['Payment_Ratio'] = np.where(df['Invoice'] == 'Paid', 1.0, 0.0)
        
    return df, notes


@st.cache_resource
def load_model():
    try:
        with open(os.path.join(OUTPUT_DIR, 'churn_prediction_model.pkl'), 'rb') as f:
            model_data = pickle.load(f)

            # Add missing keys with defaults (same as before)
            if 'model_name' not in model_data and 'model' in model_data:
                model_data['model_name'] = model_data['model'].__class__.__name__
            if 'model_display_name' not in model_data:
                model_data['model_display_name'] = model_data.get('model_name', 'Unknown Model')
            if 'best_balance_method' not in model_data:
                model_data['best_balance_method'] = 'none'

            return model_data
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run the notebook to train and save the model.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
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
    if svg_logo:
        st.markdown(f"""
            <div style="width: 380px; margin: 0 auto; padding-bottom: 10px;">
                {svg_logo}
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #94a3b8; margin-top: 10px; text-transform: uppercase; letter-spacing: 3.5px; font-weight: 500;">
                Executive Portal
            </div>
        """, unsafe_allow_html=True)
    else:
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
        if svg_logo:
            st.markdown(f"""
            <div style="text-align:center; padding: 25px 0 20px 0; background: linear-gradient(180deg, rgba(30,41,59,0.4) 0%, transparent 100%); border-bottom: 1px solid rgba(255,255,255,0.03); margin-bottom: 15px;">
                <div style="width: 80%; margin: 0 auto; padding-bottom: 5px;">
                    {svg_logo}
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 10px; color: #94a3b8; margin-top: 10px; text-transform: uppercase; letter-spacing: 3.5px; font-weight: 500;">
                    Candidate Analytics
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
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
            ("CRM Notes Analysis", ":material/call:"),
            ("Invoice Analysis", ":material/payments:"),
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
            ("<i class='fa-solid fa-users'></i>", "Enrolled & Registered", ".xlsx &nbsp;·&nbsp; 1084 rows"),
            ("<i class='fa-solid fa-clipboard-list'></i>", "CRM All Contacts",  ".xlsx &nbsp;·&nbsp; 1084 rows"),
            ("<i class='fa-solid fa-file-invoice'></i>", "Notes Processed",  ".csv &nbsp;·&nbsp; 30K+ rows"),
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
def page_overview(df, notes):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-chart-pie"></i> Executive Overview</h1>
        <p>Real-time snapshot of candidate churn status, sources, and engagement metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    total       = len(df)
    churned     = (df['churn'] == 1).sum()
    active      = total - churned
    churn_rate  = (churned / total * 100) if total > 0 else 0
    total_notes = len(notes) if notes is not None else 0

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
            <div class="kpi-icon"><i class="fa-solid fa-clipboard-list" style="color:#a78bfa"></i></div>
            <div class="kpi-title">Total CRM Notes</div>
            <div class="kpi-value" style="color:#a78bfa">{total_notes}</div>
            <div class="kpi-sub">Across all interactions</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Suggested Churn Reasons (from model outputs) ─────────
    if 'Suggested_Churn_Reason' in df.columns:
        try:
            unique_churn = df[df['churn'] == 1].drop_duplicates(subset=['Contact Id'])
            mapped_reasons = unique_churn['Suggested_Churn_Reason'].dropna().astype(str).apply(normalize_reason_label)
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
        src = df.groupby(['Source of lead', 'churn']).size().reset_index(name='Count')
        src['Status'] = src['churn'].map({0: 'Active', 1: 'Churned'})
        fig2 = px.bar(src, x='Source of lead', y='Count', color='Status',
                      color_discrete_map={'Active': COLOR_ACTIVE, 'Churned': COLOR_CHURN},
                      barmode='group', text='Count')
        fig2.update_traces(textfont_size=11, textposition='outside')
        fig2.update_layout(**theme(height=310, showlegend=True))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Course + Background + Mode ─────────
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown('<div class="section-header"><h2>By Course</h2></div>', unsafe_allow_html=True)
        course_churn = df[df['churn'] == 1]['Course'].value_counts().reset_index()
        course_churn.columns = ['Course', 'Churned']
        fig3 = px.bar(course_churn, x='Churned', y='Course', orientation='h',
                      color='Churned', color_continuous_scale=['#4c1d95','#f87171'])
        fig3.update_layout(**theme(height=280, showlegend=False,
                                   coloraxis_showscale=False))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header"><h2>Background Split</h2></div>', unsafe_allow_html=True)
        bg = df.groupby(['background', 'churn']).size().reset_index(name='Count')
        bg['Status'] = bg['churn'].map({0: 'Active', 1: 'Churned'})
        fig4 = px.pie(bg, names='background', values='Count',
                      color='background', hole=0.5,
                      color_discrete_sequence=PALETTE)
        fig4.update_layout(**theme(height=280, showlegend=True))
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.markdown('<div class="section-header"><h2>Training Mode</h2></div>', unsafe_allow_html=True)
        mode_data = df.groupby(['Mode of Program Joined', 'churn']).size().reset_index(name='Count')
        mode_data['Status'] = mode_data['churn'].map({0: 'Active', 1: 'Churned'})
        fig5 = px.bar(mode_data, x='Mode of Program Joined', y='Count', color='Status',
                      color_discrete_map={'Active': COLOR_ACTIVE, 'Churned': COLOR_CHURN},
                      barmode='stack')
        fig5.update_layout(**theme(height=280))
        st.plotly_chart(fig5, use_container_width=True)

    # ── Row 3: Role + Inferred Reason ───────────────
    col6, col7 = st.columns(2)
    with col6:
        st.markdown('<div class="section-header"><h2>Candidate Role vs Churn</h2></div>', unsafe_allow_html=True)
        ind = df.groupby(['role', 'churn']).size().reset_index(name='Count')
        ind['Status'] = ind['churn'].map({0: 'Active', 1: 'Churned'})
        fig6 = px.bar(ind, x='role', y='Count', color='Status',
                      color_discrete_map={'Active': COLOR_ACTIVE, 'Churned': COLOR_CHURN},
                      barmode='group', text='Count')
        fig6.update_traces(textfont_size=11, textposition='outside')
        fig6.update_layout(**theme(height=290))
        st.plotly_chart(fig6, use_container_width=True)

    with col7:
        st.markdown('<div class="section-header"><h2>Inferred CRM Feedback vs Churn</h2></div>', unsafe_allow_html=True)
        fb = df.groupby(['final_inferred_reason', 'churn']).size().reset_index(name='Count')
        # Limit to top 10 reasons to avoid chart clutter
        fb = fb.sort_values('Count', ascending=False).head(20)
        fb['Status'] = fb['churn'].map({0: 'Active', 1: 'Churned'})
        fig7 = px.bar(fb, x='final_inferred_reason', y='Count', color='Status',
                      color_discrete_map={'Active': COLOR_ACTIVE, 'Churned': COLOR_CHURN},
                      barmode='group', text='Count')
        fig7.update_traces(textfont_size=11, textposition='outside')
        fig7.update_layout(**theme(height=290))
        st.plotly_chart(fig7, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2 — CANDIDATE EXPLORER
# ─────────────────────────────────────────────
def page_candidate_explorer(df, notes):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-magnifying-glass"></i> Candidate Explorer</h1>
        <p>Browse, filter, and inspect individual candidate records with CRM note history.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────
    with st.expander("Filter Candidates", expanded=True):
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            sel_churn = st.selectbox("Churn Status", ["All", "Churned", "Active"], key="f_churn")
        with f2:
            sel_source = st.selectbox("Source", ["All"] + sorted(df['Source of lead'].dropna().unique().tolist()), key="f_source")
        with f3:
            sel_course = st.selectbox("Course", ["All"] + sorted(df['Course'].dropna().unique().tolist()), key="f_course")
        with f4:
            sel_mode = st.selectbox("Mode", ["All"] + sorted(df['Mode of Program Joined'].dropna().unique().tolist()), key="f_mode")
        with f5:
            sel_bg = st.selectbox("Background", ["All"] + sorted(df['background'].dropna().unique().tolist()), key="f_bg")

    fdf = df.copy()

    if sel_churn == "Churned": fdf = fdf[fdf['churn'] == 1]
    elif sel_churn == "Active": fdf = fdf[fdf['churn'] == 0]
    if sel_source != "All": fdf = fdf[fdf['Source of lead'] == sel_source]
    if sel_course != "All": fdf = fdf[fdf['Course'] == sel_course]
    if sel_mode   != "All": fdf = fdf[fdf['Mode of Program Joined']   == sel_mode]
    if sel_bg     != "All": fdf = fdf[fdf['background'] == sel_bg]

    st.markdown(f"<p style='color:#64748b; font-size:13px;'>Showing <b style='color:#a78bfa'>{len(fdf)}</b> candidates</p>", unsafe_allow_html=True)

    # ── Table ─────────────────────────────────────
    display_cols = ['Contact Id', 'Contact Name', 'Source of lead', 'Course', 'Mode of Program Joined',
                    'background', 'role', 'Status', 'Invoice', 'churn', 'Suggested_Churn_Reason']
    display_cols = [c for c in display_cols if c in fdf.columns]

    tbl = fdf[display_cols].copy()
    tbl['churn'] = tbl['churn'].map({0: 'Active', 1: 'Churned'})

    st.dataframe(
        tbl.reset_index(drop=True),
        use_container_width=True,
        height=280,
        column_config={
            "Contact Id":             st.column_config.TextColumn("ID", width="small"),
            "Contact Name":           st.column_config.TextColumn("Name"),
            "Source of lead":         st.column_config.TextColumn("Source"),
            "Mode of Program Joined": st.column_config.TextColumn("Mode"),
            "background":             st.column_config.TextColumn("Background"),
            "churn":                  st.column_config.TextColumn("Churn Status"),
            "Invoice":                st.column_config.TextColumn("Invoice Status"),
            "Suggested_Churn_Reason": st.column_config.TextColumn("AI Insight"),
        }
    )

    # ── Individual Profile ────────────────────────
    st.markdown('<div class="section-header"><h2>Candidate Profile Deep-Dive</h2></div>', unsafe_allow_html=True)
    if 'Contact Id' not in fdf.columns:
        st.info("No candidates match the current filters or 'Contact Id' is missing.")
        return
        
    candidate_ids = fdf['Contact Id'].tolist()
    if not candidate_ids:
        st.info("No candidates match the current filters.")
        return

    sel_id = st.selectbox("Select Candidate", candidate_ids, key="profile_id")
    row    = df[df['Contact Id'] == sel_id].iloc[0]
    
    # Filter notes
    cand_notes = pd.DataFrame()
    if notes is not None and not notes.empty and 'Parent ID.id' in notes.columns:
        cand_notes = notes[notes['Parent ID.id'] == sel_id].copy()

    churn_label = "CHURNED" if row.get('churn', 0) == 1 else "ACTIVE"
    churn_color = "#f87171" if row.get('churn', 0) == 1 else "#34d399"

    # ── Pre-build Notes History card content ───────────────────────
    if not cand_notes.empty:
        total_notes = len(cand_notes)
        last_note_date = cand_notes['Created Time'].max() if 'Created Time' in cand_notes.columns else "Unknown"
        notes_stats_html = f"""<div style="display:flex; gap:12px; margin-bottom:16px;">
<div style="flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:15px; text-align:center;">
<div style="font-size:28px; font-weight:800; color:#60a5fa;">{total_notes}</div>
<div style="font-size:12px; color:#94a3b8; margin-top:4px;">Total CRM Notes</div>
</div>
<div style="flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:15px; text-align:center;">
<div style="font-size:14px; font-weight:800; color:#34d399; margin-top:10px;">{str(last_note_date)[:10]}</div>
<div style="font-size:12px; color:#94a3b8; margin-top:4px;">Last Interaction</div>
</div>
</div>"""
    else:
        notes_stats_html = "<p style='color:#475569; font-style:italic;'>No CRM notes found.</p>"

    # ── Pre-build AI churn reason + recommendation HTML ────────────
    churn_insights_html = ""
    if 'Suggested_Churn_Reason' in row.index and pd.notna(row['Suggested_Churn_Reason']) and row['Suggested_Churn_Reason'] != '':
        reason_data = row['Suggested_Churn_Reason']
        rec_data = row.get('Recommended_Action', '')
        reason_text = reason_data
        rec_text = rec_data

        def clean_dict_str(obj_s):
            if isinstance(obj_s, str) and (obj_s.startswith('{') or obj_s.startswith('[')):
                import ast
                try:
                    obj = ast.literal_eval(obj_s)
                    if isinstance(obj, dict):
                        return "<br>".join([f"&bull; <b>{k}:</b> {v}" for k, v in obj.items()])
                    elif isinstance(obj, list):
                        return "<br>".join([f"&bull; {v}" for v in obj])
                except:
                    pass
            return obj_s

        reason_text = clean_dict_str(reason_text)
        rec_text = clean_dict_str(rec_text)

        churn_insights_html += f"""<div style='padding:16px; border-radius:10px; background:linear-gradient(135deg, rgba(239,68,68,0.05) 0%, rgba(239,68,68,0.02) 100%); border:1px solid rgba(239,68,68,0.2); box-shadow:0 4px 16px rgba(239,68,68,0.05); margin-bottom:12px;'>
<div style="display:flex; align-items:center; margin-bottom:8px;">
<i class="fa-solid fa-magnifying-glass-chart" style="color:#ef4444; font-size:16px; margin-right:10px;"></i>
<h4 style="margin:0; color:#f8fafc; font-family:'Inter',sans-serif; font-size:14px; font-weight:700;">AI Detected Churn Risk Factor</h4>
</div>
<div style="color:#e2e8f0; font-size:14px; line-height:1.5; margin-left:26px;">{markdown.markdown(str(reason_text))}</div>
</div>"""
        if rec_text:
            churn_insights_html += f"""<div style='padding:16px; border-radius:10px; background:linear-gradient(135deg, rgba(16,185,129,0.05) 0%, rgba(16,185,129,0.02) 100%); border:1px solid rgba(16,185,129,0.2); box-shadow:0 4px 16px rgba(16,185,129,0.05);'>
<div style="display:flex; align-items:center; margin-bottom:8px;">
<i class="fa-solid fa-wand-magic-sparkles" style="color:#10b981; font-size:16px; margin-right:10px;"></i>
<h4 style="margin:0; color:#f8fafc; font-family:'Inter',sans-serif; font-size:14px; font-weight:700;">AI Retention Recommendation</h4>
</div>
<div style="color:#e2e8f0; font-size:14px; line-height:1.5; margin-left:26px;">{markdown.markdown(str(rec_text))}</div>
</div>"""


    # ── Render all three columns ────────────────────────────────────
    pc1, pc2, pc3 = st.columns([1.2, 1.2, 1.6])
    card_style = "height:100%; min-height:500px; display:flex; flex-direction:column; justify-content:space-between;"

    # Add raw CRM notes to UI
    raw_notes_html = "<div style='max-height: 250px; overflow-y: auto; padding-right: 8px; margin-top: 10px;'>"
    if not cand_notes.empty:
        for idx, note_row in cand_notes.sort_values(by='Created Time', ascending=False).iterrows():
            date_str = str(note_row.get('Created Time', ''))[:16]
            exec_name = str(note_row.get('Note Owner', 'Unknown Exec'))
            note_content = str(note_row.get('Note Content', 'No content'))
            raw_notes_html += f"""
            <div style="background:rgba(0,0,0,0.2); border:1px solid rgba(255,255,255,0.05); padding:10px; border-radius:6px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:11px; color:#94a3b8;">
                    <span><b><i class="fa-solid fa-user-tie"></i> {exec_name}</b></span>
                    <span>{date_str}</span>
                </div>
                <div style="font-size:13px; color:#e2e8f0; line-height:1.4;">{note_content}</div>
            </div>"""
    else:
        raw_notes_html += "<div style='color:#64748b; font-size:13px;'>No specific CRM notes logged.</div>"
    raw_notes_html += "</div>"

    with pc1:

        st.markdown(f"""<div class="candidate-card" style="{card_style}">
<div>
<div style="font-size:16px; font-weight:800; color:#f8fafc; margin-bottom:16px; text-transform:uppercase; letter-spacing:1.5px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px;"><i class="fa-regular fa-id-card" style="color:#6366f1; margin-right:8px;"></i> Personal Info</div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Name:</span> <b style="color:#e2e8f0;">{row.get('Contact Name','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">ID:</span> <b style="color:#a78bfa;">{row['Contact Id']}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Background:</span> <b style="color:#e2e8f0;">{row.get('background','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Role:</span> <b style="color:#e2e8f0;">{row.get('role','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Experience:</span> <b style="color:#e2e8f0;">{row.get('Experience','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Gender:</span> <b style="color:#e2e8f0;">{row.get('Gender','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Source:</span> <b style="color:#e2e8f0;">{row.get('Source of lead','N/A')}</b></div>
</div>
<div style="margin-top:14px; padding:10px 14px; border-radius:8px; background:rgba(0,0,0,0.2); text-align:center;">
<span style="color:{churn_color}; font-weight:800; font-size:16px;">{churn_label}</span>
</div>
</div>""", unsafe_allow_html=True)

    with pc2:
        inv_status = row.get('Invoice', 'No Invoice')
        inv_color = "#34d399" if inv_status == 'Paid' else "#f87171"
        st.markdown(f"""<div class="candidate-card" style="{card_style}">
<div>
<div style="font-size:16px; font-weight:800; color:#f8fafc; margin-bottom:16px; text-transform:uppercase; letter-spacing:1.5px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px;"><i class="fa-solid fa-graduation-cap" style="color:#34d399; margin-right:8px;"></i> Course &amp; Status</div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Course:</span> <b style="color:#e2e8f0;">{row.get('Course','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Program Joined:</span> <b style="color:#e2e8f0;">{row.get('Program Joined','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Mode:</span> <b style="color:#e2e8f0;">{row.get('Mode of Program Joined','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Location:</span> <b style="color:#e2e8f0;">{row.get('Program Location','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Current Status:</span> <b style="color:#fbbf24;">{row.get('Status','N/A')}</b></div>
<div style="margin-bottom:10px;"><span style="color:#64748b;">Invoice Status:</span> <b style="color:{inv_color};">{inv_status}</b></div>
</div>
</div>""", unsafe_allow_html=True)

    with pc3:
        st.markdown(f"""<div class="candidate-card" style="{card_style}; justify-content:flex-start;">
<div style="font-size:16px; font-weight:800; color:#f8fafc; margin-bottom:16px; text-transform:uppercase; letter-spacing:1.5px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px;"><i class="fa-solid fa-clipboard" style="color:#f43f5e; margin-right:8px;"></i> CRM Notes History</div>
{notes_stats_html}
{churn_insights_html}
{raw_notes_html}
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 3 — CRM NOTES ANALYSIS
# ─────────────────────────────────────────────
def page_notes_analysis(df, notes):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-clipboard"></i> CRM Notes Analysis</h1>
        <p>Understand interaction volume, executive engagement, and inferred candidate sentiments.</p>
    </div>
    """, unsafe_allow_html=True)

    if notes is None or notes.empty:
        st.warning("No CRM Notes data available.")
        return

    # ── KPIs ─────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    total_notes = len(notes)
    unique_cands = notes['Parent ID.id'].nunique()
    avg_notes_per_cand = total_notes / unique_cands if unique_cands > 0 else 0
    unique_owners = notes['Note Owner'].nunique() if 'Note Owner' in notes.columns else 0

    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-clipboard-list" style="color:#a78bfa"></i></div>
            <div class="kpi-title">Total Notes</div>
            <div class="kpi-value" style="color:#60a5fa">{total_notes}</div>
            <div class="kpi-sub">Across all CRM records</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-users" style="color:#34d399"></i></div>
            <div class="kpi-title">Candidates Reached</div>
            <div class="kpi-value" style="color:#34d399">{unique_cands}</div>
            <div class="kpi-sub">With at least one note</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-rotate"></i></div>
            <div class="kpi-title">Avg Notes/Candidate</div>
            <div class="kpi-value" style="color:#a78bfa">{avg_notes_per_cand:.1f}</div>
            <div class="kpi-sub">Interaction frequency</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-user-tie"></i></div>
            <div class="kpi-title">Active Executives</div>
            <div class="kpi-value" style="color:#fbbf24">{unique_owners}</div>
            <div class="kpi-sub">Logging notes</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header"><h2>Notes per Executive</h2></div>', unsafe_allow_html=True)
        if 'Note Owner' in notes.columns:
            exec_notes = notes.groupby('Note Owner').size().reset_index(name='Notes').sort_values('Notes', ascending=True)
            exec_notes = exec_notes.tail(15)
            fig = px.bar(exec_notes, x='Notes', y='Note Owner', orientation='h',
                         color='Notes', color_continuous_scale=['#4c1d95','#6366f1','#06b6d4'],
                         text='Notes')
            fig.update_traces(textfont_size=11, textposition='outside')
            fig.update_layout(**theme(height=310, showlegend=False, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Note Owner data not found.")

    with col2:
        st.markdown('<div class="section-header"><h2>Notes Volume: Active vs Churned Candidates</h2></div>', unsafe_allow_html=True)
        if 'Parent ID.id' in notes.columns and 'Contact Id' in df.columns:
            merged = df[['Contact Id','churn']].merge(notes, left_on='Contact Id', right_on='Parent ID.id', how='right')
            merged['Status'] = merged['churn'].map({0:'Active', 1:'Churned'}).fillna('Unknown')
            dur_by_status = merged.groupby('Status').size().reset_index(name='Notes Count')
            fig2 = px.pie(dur_by_status, names='Status', values='Notes Count',
                          color='Status', color_discrete_map={'Active':COLOR_ACTIVE,'Churned':COLOR_CHURN, 'Unknown':'#64748b'}, hole=0.5)
            fig2.update_layout(**theme(height=310, showlegend=True))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Mapping between candidates and notes not available.")

    if 'Created Time' in notes.columns:
        st.markdown('<div class="section-header"><h2><i class="fa-solid fa-calendar-days"></i> CRM Note Activity Timeline</h2></div>', unsafe_allow_html=True)
        try:
            notes['Date'] = pd.to_datetime(notes['Created Time']).dt.date
            timeline = notes.groupby('Date').size().reset_index(name='Notes')
            timeline['Date'] = pd.to_datetime(timeline['Date'])
            fig4 = go.Figure(go.Scatter(
                x=timeline['Date'], y=timeline['Notes'],
                mode='lines+markers',
                line=dict(color='#6366f1', width=2.5),
                marker=dict(size=7, color='#a78bfa'),
                fill='tozeroy', fillcolor='rgba(255,255,255,0.1)',
                hovertemplate='<b>%{x|%d %b %Y}</b><br>Notes: %{y}<extra></extra>'
            ))
            fig4.update_layout(**theme(
                height=260,
                xaxis=dict(title='Date', gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False),
                yaxis=dict(title='Number of Notes', gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False),
            ))
            st.plotly_chart(fig4, use_container_width=True)
        except Exception:
            st.warning("Could not parse 'Created Time' for timeline.")

# ─────────────────────────────────────────────
# PAGE 4 — PAYMENT ANALYSIS
# ─────────────────────────────────────────────
def page_payment_analysis(df):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-money-bill-wave"></i> Invoice & Status Analysis</h1>
        <p>Invoice generation status and collection trends across courses.</p>
    </div>
    """, unsafe_allow_html=True)

    if 'Invoice' not in df.columns:
        st.warning("Invoice data not available.")
        return

    df_inv = df.copy()
    df_inv['Invoice'] = df_inv['Invoice'].fillna('No Invoice').astype(str).str.title().str.strip()
    
    total_cands = len(df_inv)
    paid_count  = len(df_inv[df_inv['Invoice'] == 'Paid'])
    sent_count  = len(df_inv[df_inv['Invoice'] == 'Sent'])
    no_inv      = len(df_inv[df_inv['Invoice'].isin(['No', 'No Invoice', 'Nan'])])
    paid_rate   = paid_count / total_cands * 100 if total_cands > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-users"></i></div>
            <div class="kpi-title">Total Candidates</div>
            <div class="kpi-value" style="color:#60a5fa; font-size:26px;">{total_cands}</div>
            <div class="kpi-sub">Total records</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-check-circle" style="color:#34d399"></i></div>
            <div class="kpi-title">Paid Invoices</div>
            <div class="kpi-value" style="color:#34d399; font-size:26px;">{paid_count}</div>
            <div class="kpi-sub">{paid_rate:.1f}% paid rate</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-paper-plane" style="color:#fbbf24"></i></div>
            <div class="kpi-title">Sent (Pending)</div>
            <div class="kpi-value" style="color:#fbbf24; font-size:26px;">{sent_count}</div>
            <div class="kpi-sub">Awaiting payment</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon"><i class="fa-solid fa-ban" style="color:#f87171"></i></div>
            <div class="kpi-title">No Invoice</div>
            <div class="kpi-value" style="color:#f87171; font-size:26px;">{no_inv}</div>
            <div class="kpi-sub">Not yet generated</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header"><h2>Invoice Status Distribution</h2></div>', unsafe_allow_html=True)
        inv_counts = df_inv['Invoice'].value_counts().reset_index()
        inv_counts.columns = ['Invoice Status', 'Count']
        fig = px.pie(inv_counts, names='Invoice Status', values='Count', hole=0.4,
                     color_discrete_sequence=['#34d399', '#f87171', '#fbbf24', '#a78bfa', '#94a3b8'])
        fig.update_layout(**theme(height=300, showlegend=True))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h2>Invoice Status by Course</h2></div>', unsafe_allow_html=True)
        pm = df_inv.groupby(['Course','Invoice']).size().reset_index(name='Count')
        fig2 = px.bar(pm, x='Course', y='Count', color='Invoice',
                      color_discrete_sequence=['#34d399', '#fbbf24', '#f87171', '#94a3b8'],
                      barmode='stack')
        fig2.update_layout(**theme(height=300, showlegend=True))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header"><h2>Churn Distribution by Invoice Status</h2></div>', unsafe_allow_html=True)
    inv_churn = df_inv.groupby(['Invoice', 'churn']).size().reset_index(name='Count')
    inv_churn['Status'] = inv_churn['churn'].map({0:'Active', 1:'Churned'})
    fig3 = px.bar(inv_churn, x='Invoice', y='Count', color='Status',
                  color_discrete_map={'Active':COLOR_ACTIVE,'Churned':COLOR_CHURN},
                  barmode='group', text='Count')
    fig3.update_traces(textfont_size=12, textposition='outside')
    fig3.update_layout(**theme(height=360))
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 5 — LIVE PREDICTOR
# ─────────────────────────────────────────────
def page_live_predictor(df, model_data):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-robot"></i> Live Churn Predictor</h1>
        <p>Dynamic AI predictor based on the active model schema. Fill in candidate details to assess risk.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_data is None:
        st.error("Could not load churn_prediction_model.pkl")
        return

    available_models = model_data.get('available_models', {})

    # Check for dict specifically
    if not isinstance(available_models, dict) or (hasattr(available_models, 'empty') and available_models.empty):
        available_models = {model_data.get('model_display_name', 'Default Model'): model_data.get('model')}

    #if not available_models:
    #    # Fallback to single model if available_models is missing (older model format)
    #    available_models = {model_data.get('model_display_name', 'Default Model'): model_data.get('model')}
    
    # Model Selection UI
    st.markdown('<div class="section-header"><h2>Candidate Details & AI Settings</h2></div>', unsafe_allow_html=True)
    
    llm_options = ["Gemini 2.5 Flash", "Groq (Llama 3)", "Hugging Face (Mistral)"]
    ml_options = list(available_models.keys())
    all_options = llm_options + ml_options
    
    col_ai, _ = st.columns([1, 2])
    with col_ai:
        selected_model_name = st.selectbox(
            "Select AI Model",
            options=all_options,
            help="Choose which trained algorithm or AI to use for the prediction."
        )
    
    is_llm = selected_model_name in llm_options
    if not is_llm:
        model = available_models[selected_model_name]
    final_model = model_data.get('model')
    feature_columns = model_data.get('feature_columns', [])
    preprocessor = model_data.get('preprocessor', {})
    categorical_features = model_data.get('categorical_features', [])
    numerical_features  = model_data.get('numerical_features', [])
    label_encoders  = model_data.get('label_encoders', {})
    scaler          = model_data.get('scaler', None)
    balance_method = model_data.get('best_balance_method', 'none')

    st.markdown('<div class="section-header"><h2>Candidate Details</h2></div>', unsafe_allow_html=True)
    st.markdown(
        f"**Model:** {model_data.get('model_display_name', 'Unknown')}  •  **Balancing:** {format_balance_method(balance_method)}")

    df = df[df['churn'].isin([0, 1])].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", sorted(df['Gender'].unique()), key="p_gender")
        source = st.selectbox("Source of lead", sorted(df['Source of lead'].unique()), key="p_source")
        education = st.selectbox("Course", sorted(df['Course'].unique()), key="p_edu")
        semester = st.number_input("Semester", 0, 10, 3, key="p_sem")
        background = st.selectbox("background", sorted(df['background'].unique()), key="p_bg")

    with col2:
        stream = st.selectbox("Program_Name", sorted(df['Program_Name'].unique()), key="p_stream")
        role = st.selectbox("role", sorted(df['role'].unique()), key="p_role")
        mode = st.selectbox("Mode of Program Joined", sorted(df['Mode of Program Joined'].unique()), key="p_mode")
        track_interested = st.selectbox("Track Interested", sorted(df['Track Interested'].unique()), key="p_track")

    with col3:
        status = st.selectbox("final_inferred_status", sorted(df['final_inferred_status'].unique()), key="p_cs")
        experience = st.number_input("Experience (years)", 0, 30, 3, key="p_exp")
        year_of_graduation = st.number_input("Year of Graduation (0 if not graduated)", min_value=0, max_value=2050,
                                             value=2024)
        batch_assigned_to = st.selectbox("Batch Assigned to",sorted(df['Batch Assigned to'].unique()), key="p_ba")

    st.markdown('<div class="section-header"><h2>Call Inputs</h2></div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        test_taken = st.checkbox("Attended an assessment", value=True, key="p_tt")
    with sc2:
        followup_email_sent = st.checkbox("Followup after call", value=False, key="p_fe")
    with sc3:
        invoice_generated = st.checkbox("Payment Done", value=False, key="p_ig")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        not_interested = st.checkbox("No Interest in Course", value=True, key="p_ni")
    with sc2:
        unreachable_not_connected = st.checkbox("No Response / Unreachable", value=False, key="p_ur")
    with sc3:
        joined_competitor = st.checkbox("Joined in another institution", value=False, key="p_jc")
    with sc4:
        financial_issue = st.checkbox("Course fees not affordable", value=True, key="p_fs")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        already_working = st.checkbox("Got placed", value=True, key="p_aw")
    with sc2:
        looking_for_job = st.checkbox("Job hunting, not internship", value=False, key="p_lj")
    with sc4:
        no_notes_provided = st.checkbox("No notes after call", value=False, key="p_nn")
    with sc3:
        decision_pending = st.checkbox("Technical Discussion", value=True, key="p_dp")

    # Free-text call remarks and optional transcript for live inference
    call_remarks = st.text_area("Call Remarks (optional)", value="", max_chars=1000,
                                placeholder="Enter recent call remarks or notes...", key="p_remarks")
    call_transcript = st.text_area("Call Transcript (optional)", value="", max_chars=2000,
                                   placeholder="Paste full call transcript to improve AI churn reason extraction.",
                                   key="p_transcript")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Churn Risk", use_container_width=True, type="primary"):


        input_data = {'Experience': experience,
            'Semester': semester,
            'Year of Graduation': year_of_graduation,
            'Test': int(test_taken),
            'Followup Email': int(followup_email_sent),

            # Categorical features
            'Source of lead': source,
            'Course': education,
            'background': background, # Placeholder, ideally derived dynamically from course
            'role': role,# Placeholder, ideally derived dynamically from experience/semester/grad_year
            #'Program_Name': stream,
            'Track Interested': track_interested,
            'Mode of Program Joined': mode,
            'Gender': gender
        }

        if not is_llm:
            input_df = pd.DataFrame([input_data])

            # The `preprocessor` expects a DataFrame with the original feature names.
            # `input_df` contains these. `feature_columns` from the saved model refer to the columns *after* preprocessing.
            processed_input = preprocessor.transform(input_df)

            # Convert to DataFrame with correct column names for the model if needed
            processed_input_df = pd.DataFrame(
                processed_input.toarray() if hasattr(processed_input, 'toarray') else processed_input,
                columns=feature_columns)

            # STEP 3: Align with training features - THIS IS CRITICAL
            # Remove columns that aren't in training
            cols_to_drop = [col for col in processed_input_df.columns if col not in feature_columns]
            if cols_to_drop:
                processed_input_df = processed_input_df.drop(columns=cols_to_drop)

            # Align columns - add missing columns with 0 and remove extra ones
            for col in feature_columns:
                if col not in processed_input_df.columns:
                    processed_input_df[col] = 0

            processed_input_df = processed_input_df[feature_columns]  # Ensure order and presence

            # Scale numerical features
            #processed_input[numerical_feat] = scaler.transform(processed_input[numerical_feat])

            # Add any missing model features with safe defaults
            #for col in feature_columns:
            #    if col not in input_df.columns:
            #        if col in numerical_feat:
            #            input_df[col] = 0
            #        else:
            #            input_df[col] = 'Unknown'

        else:
            processed_input = None


        #X[numerical_feat] = scaler.transform(X[numerical_feat].values.reshape(1, -1))
                # If scaler fails, we just continue with raw values

        try:
            if is_llm:
                from llm_integration import get_llm_prediction
                with st.spinner(f"Querying {selected_model_name} AI Strategist..."):
                    llm_result = get_llm_prediction(selected_model_name, input_data)
                    
                prob = float(llm_result.get("churn_probability", 50.0)) / 100.0
                pred = 1 if prob >= 0.5 else 0
                llm_reason = llm_result.get("reason", "No reason provided by LLM.")
                llm_retention = llm_result.get("retention_strategy", "No strategy provided.")
                model_display = selected_model_name
            else:
                # Scikit-learn normal pipeline
                # Make prediction
                pred_raw = final_model.predict(processed_input_df)
                prob_raw = final_model.predict_proba(processed_input_df)[:, 1]

                # Extract scalar values from numpy arrays
                pred = int(np.asarray(pred_raw).flatten()[0])
                prob = float(np.asarray(prob_raw).flatten()[0])

                # Clamp probability between 0 and 1
                prob = max(0.0, min(1.0, prob))

                llm_reason = "Based on machine learning feature importance patterns."
                llm_retention = "Follow standard operational procedures."
                model_display = selected_model_name

            # Result Display
            r1, r2, r3 = st.columns([1.2, 1.2, 1.6])
            with r1:
                if pred == 1:
                    st.markdown(f"""
                    <div class="prediction-box-churn" style="height:100%;">
                        <div class="pred-label" style="color:#f87171;"><i class="fa-solid fa-circle-xmark"></i></div>
                        <div style="font-size:24px; font-weight:800; color:#f87171; margin-bottom:8px;">HIGH CHURN RISK</div>
                        <div class="pred-sub">This candidate is likely to NOT join training</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box-safe" style="height:100%;">
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
                action_req = '<i class="fa-solid fa-triangle-exclamation" style="color:#fbbf24"></i> <b style="color:#fbbf24;">Action Required:</b> Immediate intervention.' if pred == 1 else '<i class="fa-solid fa-circle-check" style="color:#34d399"></i> <b style="color:#34d399;">On Track:</b> Monitor.'
                
                st.markdown(f"""<div class="candidate-card" style="margin-top:0; height:100%; display:flex; flex-direction:column;">
<div style="font-size:13px; font-weight:700; color:#94a3b8; margin-bottom:10px; text-transform:uppercase; letter-spacing:1px;">AI Strategist Assessment</div>
<div style="font-size:13px; color:#e2e8f0; margin-bottom:8px;">
<span style="color:#64748b; font-weight:600;">Reasoning:</span><br>
<i>"{llm_reason}"</i>
</div>
<div style="font-size:13px; color:#38bdf8; margin-bottom:10px; flex-grow:1;">
<span style="color:#64748b; font-weight:600;">Retention Strategy:</span><br>
<b>{llm_retention}</b>
</div>
<hr style="border-color:rgba(255,255,255,0.2); margin:8px 0;">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div style="font-size:11px; color:#64748b;">Using {model_display}</div>
<div style="font-size:12px;">{action_req}</div>
</div>
</div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}\\n\\nPlease ensure all required inputs are provided.")

# ─────────────────────────────────────────────# ─────────────────────────────────────────────
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
        estimator_value = getattr(model, 'n_estimators', None) or getattr(model, 'n_estimators_', 'N/A')
        st.markdown(f"""<div class=\"kpi-card\"><div class=\"kpi-icon\"><i class="fa-solid fa-tree"></i></div>
                <div class=\"kpi-title\">Estimators</div>
                <div class=\"kpi-value\" style=\"color:#60a5fa;\">{estimator_value}</div>
                <div class=\"kpi-sub\">Decision trees</div></div>""", unsafe_allow_html=True)
    with i3:
        max_depth_value = getattr(model, 'max_depth', 'N/A')
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

    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-chart-bar"></i> Model Performance Tracker</h1>
        <p>Live metrics from the pipeline evaluation outputs.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Churn Reason Analysis ─────────────────────
    st.markdown(
        '<div class="section-header"><h2><i class="fa-solid fa-clipboard-list"></i> Why Are Candidates Churning? — Reason Analysis</h2></div>',
        unsafe_allow_html=True)

    churned_df = df[df['churn'] == 1].copy()
    active_df = df[df['churn'] == 0].copy()

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown("""<div class="candidate-card">
                           <div style="font-size:13px; font-weight:700; color:#f87171; margin-bottom:12px;"><i class="fa-solid fa-circle-xmark"></i> No notes</div>""",
                    unsafe_allow_html=True)
        # Filter for 'not interested' only
        no_interest_mask = churned_df['final_inferred_reason'].str.lower().str.contains('no notes provided', na=False)
        no_interest_df = churned_df[no_interest_mask]
        count = len(no_interest_df)
        pct = count / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
                <div style="font-size:32px; font-weight:800; color:#fbbf24;">{int(count)}</div>
                <div style="color:#64748b; font-size:13px;">churned had no interest ({pct:.0f}%)</div>
                <div style="margin-top:10px; font-size:12px; color:#475569;">Churned candidates made initial payment - but no longer interested, they may have attended or not attended the induction session.</div>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("""<div class="candidate-card">
                           <div style="font-size:13px; font-weight:700; color:#fbbf24; margin-bottom:12px;"><i class="fa-solid fa-phone-slash"></i> No Response Pattern</div>""",
                    unsafe_allow_html=True)
        # Filter for 'not interested' only
        no_resp_mask = churned_df['final_inferred_reason'].str.lower().str.contains('unreachable/not connected', na=False)
        no_resp_df = churned_df[no_resp_mask]
        no_resp = len(no_resp_df)
        pct2 = no_resp / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
                <div style="font-size:32px; font-weight:800; color:#fbbf24;">{int(no_resp)}</div>
                <div style="color:#64748b; font-size:13px;">churned had no-response calls ({pct2:.0f}%)</div>
                <div style="margin-top:10px; font-size:12px; color:#475569;">Unreachable candidates rarely convert — early escalation is critical.</div>
            </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown("""<div class="candidate-card">
                           <div style="font-size:13px; font-weight:700; color:#10b981; margin-bottom:12px;"><i class="fa-solid fa-money-bill-wave"></i> Financial Issue</div>""",
                    unsafe_allow_html=True)
        no_fin_mask = churned_df['final_inferred_reason'].str.lower().str.contains('financial issue', na=False)
        no_fin_df = churned_df[no_fin_mask]
        fin_issue = len(no_fin_df)
        pct_fin = fin_issue / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
            <div style="font-size:32px; font-weight:800; color:#f87171;">{int(fin_issue)}</div>
            <div style="color:#64748b; font-size:13px;">Churned with payment discussion ({pct_fin:.0f}%)</div>
            <div style="margin-top:10px; font-size:12px; color:#475569;">Candidates who had financial concerns or payment-related discussions before churning.</div>
            </div>""", unsafe_allow_html=True)

    with r4:
        st.markdown("""<div class="candidate-card">
                           <div style="font-size:13px; font-weight:700; color:#3b82f6; margin-bottom:12px;"><i class="fa-solid fa-graduation-cap"></i> Joined Another Institution</div>""",
                    unsafe_allow_html=True)
        no_join_mask = churned_df['final_inferred_reason'].str.lower().str.contains('joined competitor',
                                                                                    na=False)
        no_join_df = churned_df[no_join_mask]
        joined_competitor = len(no_join_df)
        pct_joined = joined_competitor / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
            <div style="font-size:32px; font-weight:800; color:#f87171;">{int(joined_competitor)}</div>
            <div style="color:#64748b; font-size:13px;">Churned to join another institute ({pct_joined:.0f}%)</div>
            <div style="margin-top:10px; font-size:12px; color:#475569;">Candidates who chose to enroll at a competitor institution instead.</div>
            </div>""", unsafe_allow_html=True)


    eval_path = os.path.join(OUTPUT_DIR, "model_evaluation_results.csv")
    feat_path = os.path.join(OUTPUT_DIR, "feature_importance_report.csv")
    
    if os.path.exists(eval_path):
        eval_df = pd.read_csv(eval_path)
        st.markdown("## Model Evaluation Summary")

        # Normalize column names for consistent access
        eval_df.columns = eval_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

        # Extract metrics
        model_name = eval_df['model'].iloc[0] if 'model' in eval_df.columns else model_data.get('model_display_name',
                                                                                                'Best Model')
        accuracy = eval_df['accuracy'].iloc[0] if 'accuracy' in eval_df.columns else None
        f1_score = eval_df['f1_score'].iloc[0] if 'f1_score' in eval_df.columns else (
            eval_df['f1 score'].iloc[0] if 'f1 score' in eval_df.columns else None)
        precision = eval_df['precision'].iloc[0] if 'precision' in eval_df.columns else None
        recall = eval_df['recall'].iloc[0] if 'recall' in eval_df.columns else None
        roc_auc = eval_df['roc_auc'].iloc[0] if 'roc_auc' in eval_df.columns else (
            eval_df['roc-auc'].iloc[0] if 'roc-auc' in eval_df.columns else None)
        training_time = eval_df['training_time'].iloc[0] if 'training_time' in eval_df.columns else None

        # ── Key Metrics Cards ──
        st.markdown("### Key Performance Indicators")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        # Accuracy Card
        with metric_col1:
            acc_display = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
            acc_color = "#34d399" if (accuracy and accuracy >= 0.85) else (
                "#fbbf24" if (accuracy and accuracy >= 0.70) else "#f87171")
            st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {acc_color};">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value" style="color:{acc_color};">{acc_display}</div>
                    <div class="metric-sub">Classification accuracy</div>
                </div>
                """, unsafe_allow_html=True)

        # F1 Score Card
        with metric_col2:
            f1_display = f"{f1_score * 100:.2f}%" if f1_score is not None else "N/A"
            f1_color = "#34d399" if (f1_score and f1_score >= 0.80) else (
                "#fbbf24" if (f1_score and f1_score >= 0.60) else "#f87171")
            st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {f1_color};">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value" style="color:{f1_color};">{f1_display}</div>
                    <div class="metric-sub">Harmonic mean of precision & recall</div>
                </div>
                """, unsafe_allow_html=True)

        # Precision Card
        with metric_col3:
            prec_display = f"{precision * 100:.2f}%" if precision is not None else "N/A"
            prec_color = "#34d399" if (precision and precision >= 0.80) else (
                "#fbbf24" if (precision and precision >= 0.60) else "#f87171")
            st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {prec_color};">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value" style="color:{prec_color};">{prec_display}</div>
                    <div class="metric-sub">True positive rate</div>
                </div>
                """, unsafe_allow_html=True)

        # Recall Card
        with metric_col4:
            rec_display = f"{recall * 100:.2f}%" if recall is not None else "N/A"
            rec_color = "#34d399" if (recall and recall >= 0.80) else (
                "#fbbf24" if (recall and recall >= 0.60) else "#f87171")
            st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {rec_color};">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value" style="color:{rec_color};">{rec_display}</div>
                    <div class="metric-sub">Sensitivity / Hit rate</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Secondary Metrics Row ──
        sec_col1, sec_col2, sec_col3 = st.columns(3)

        # ROC-AUC Card
        with sec_col1:
            auc_display = f"{roc_auc * 100:.2f}%" if roc_auc is not None else "N/A"
            auc_color = "#34d399" if (roc_auc and roc_auc >= 0.85) else (
                "#fbbf24" if (roc_auc and roc_auc >= 0.70) else "#f87171")
            st.markdown(f"""
                <div class="metric-card secondary" style="border-left: 4px solid {auc_color};">
                    <div class="metric-label">ROC-AUC</div>
                    <div class="metric-value" style="color:{auc_color};">{auc_display}</div>
                    <div class="metric-sub">Area under ROC curve</div>
                </div>
                """, unsafe_allow_html=True)

        # Training Time Card
        with sec_col2:
            if training_time is not None:
                if training_time < 60:
                    time_display = f"{training_time:.2f}s"
                elif training_time < 3600:
                    time_display = f"{training_time / 60:.2f}m"
                else:
                    time_display = f"{training_time / 3600:.2f}h"
            else:
                time_display = "N/A"
            st.markdown(f"""
                <div class="metric-card secondary">
                    <div class="metric-label">Training Time</div>
                    <div class="metric-value" style="color:#06b6d4;">{time_display}</div>
                    <div class="metric-sub">Model training duration</div>
                </div>
                """, unsafe_allow_html=True)

        # Training Data Info
        with sec_col3:
            if model_data.get('training_data_shape'):
                shape = model_data.get('training_data_shape')
                st.markdown(f"""
                    <div class="metric-card secondary">
                        <div class="metric-label">Training Data</div>
                        <div class="metric-value" style="color:#6366f1;">{shape[0]:,}</div>
                        <div class="metric-sub">{shape[1]} features</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="metric-card secondary">
                        <div class="metric-label">Training Data</div>
                        <div class="metric-value" style="color:#6366f1;">N/A</div>
                        <div class="metric-sub">Not available</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")


        # ── Detailed Metrics Table ──
        st.markdown("### Detailed Metrics Breakdown")

        # Format the dataframe for display
        display_df = eval_df.copy()

        # Format numeric columns to percentage where appropriate
        for col in ['accuracy', 'f1_score', 'f1 score', 'precision', 'recall', 'roc_auc', 'roc-auc']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x * 100:.2f}%" if isinstance(x, (int, float)) else str(x))

        # Format training time
        if 'training_time' in display_df.columns:
            display_df['training_time'] = display_df['training_time'].apply(
                lambda x: f"{x:.3f}s" if isinstance(x, (int, float)) else str(x)
            )

        # Create styled HTML table
        def create_styled_table(df, table_id):
            headers = df.columns.tolist()

            html = f"""
                <style>
                    #{table_id} {{
                        width: 100%;
                        border-collapse: collapse;
                        font-family: 'Inter', sans-serif;
                        font-size: 14px;
                        margin: 10px 0;
                    }}
                    #{table_id} thead {{}}
                    #{table_id} th {{
                        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
                        color: #e2e8f0;
                        padding: 14px 18px;
                        text-align: left;
                        font-weight: 600;
                        font-size: 12px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        border-bottom: 2px solid #4f46e5;
                    }}
                    #{table_id} td {{
                        padding: 12px 18px;
                        border-bottom: 1px solid #334155;
                        color: #cbd5e1;
                    }}
                    #{table_id} tr {{ background: rgba(30, 41, 59, 0.5); }}
                    #{table_id} tr:hover {{ background: rgba(99, 102, 241, 0.15); }}
                    #{table_id} tr:last-child td {{ border-bottom: none; }}
                    .highlight-cell {{
                        font-weight: 700;
                        color: #06b6d4;
                    }}
                </style>
                <table id="{table_id}">
                    <thead>
                        <tr>{"".join([f'<th>{h.replace("_", " ").title()}</th>' for h in headers])}</tr>
                    </thead>
                    <tbody>
                """

            for _, row in df.iterrows():
                html += "<tr>"
                for col, val in zip(headers, row):
                    cell_class = 'highlight-cell' if col in ['accuracy', 'f1_score', 'f1 score'] and '%' in str(
                        val) else ''
                    html += f'<td class="{cell_class}">{val}</td>'
                html += "</tr>"

            html += "</tbody></table>"
            return html

        st.markdown(create_styled_table(display_df, "metrics-table"), unsafe_allow_html=True)

        st.markdown("---")

        # ── Model Configuration Info ──
        st.markdown("### Model Configuration")

        config_col1, config_col2, config_col3, config_col4 = st.columns(4)

        with config_col1:
            st.markdown(f"""
                <div class="config-card">
                    <div class="config-icon"><i class="fa-solid fa-brain"></i></div>
                    <div class="config-label">Model</div>
                    <div class="config-value">{model_name}</div>
                </div>
                """, unsafe_allow_html=True)

        with config_col2:
            balance = model_data.get('balance_method', 'N/A')
            st.markdown(f"""
                <div class="config-card">
                    <div class="config-icon"><i class="fa-solid fa-scale-balanced"></i></div>
                    <div class="config-label">Balance Method</div>
                    <div class="config-value">{format_balance_method(balance)}</div>
                </div>
                """, unsafe_allow_html=True)

        with config_col3:
            n_features = model_data.get('training_data_shape', (0, 0))[1]
            st.markdown(f"""
                <div class="config-card">
                    <div class="config-icon"><i class="fa-solid fa-layer-group"></i></div>
                    <div class="config-label">Features</div>
                    <div class="config-value">{n_features}</div>
                </div>
                """, unsafe_allow_html=True)

        with config_col4:
            class_dist = model_data.get('class_distribution_train', {})
            if class_dist:
                ratio = class_dist.get(1, 0) / class_dist.get(0, 1) if class_dist.get(0, 0) > 0 else 0
                st.markdown(f"""
                    <div class="config-card">
                        <div class="config-icon"><i class="fa-solid fa-chart-pie"></i></div>
                        <div class="config-label">Class Ratio</div>
                        <div class="config-value">{ratio:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="config-card">
                        <div class="config-icon"><i class="fa-solid fa-chart-pie"></i></div>
                        <div class="config-label">Class Ratio</div>
                        <div class="config-value">N/A</div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
         st.error(" Evaluation results not found")


    #else:
    #    st.info("Evaluation results not found.")
        
    if os.path.exists(feat_path):
        feat_df = pd.read_csv(feat_path).head(15)
        st.markdown("### Top 15 Feature Importances")
        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale=['#4c1d95','#6366f1','#06b6d4'])
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.update_layout(**theme(height=400, showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance report not found.")


def page_profile():
    st.markdown(
        '<div class="page-header"><h1><i class="fa-solid fa-user-circle"></i> Executive Profile</h1><p>Manage your account settings and personal info.</p></div>',
        unsafe_allow_html=True)

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
            df, notes = load_data()
            df, notes = preprocess(df, notes)
        except Exception as e:
            st.error(f"Could not load data files: {e}")
            st.stop()

    model_path = os.path.join(OUTPUT_DIR, "churn_prediction_model.pkl")
    model_modified_time = os.path.getmtime(model_path) if os.path.exists(model_path) else None
    model_data = load_model()

    if   "Overview"            in page: page_overview(df, notes)
    elif "Candidate Explorer"  in page: page_candidate_explorer(df, notes)
    elif "CRM Notes Analysis"  in page: page_notes_analysis(df, notes)
    elif "Invoice Analysis"    in page: page_payment_analysis(df)
    elif "Predictor"           in page: page_live_predictor(df, model_data)
    elif "Model Performance"   in page: page_model_performance(df, model_data)


if __name__ == "__main__":
    main()
