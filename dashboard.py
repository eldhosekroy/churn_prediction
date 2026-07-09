"""
AI-Powered Candidate Churn Prediction & Reason Analysis Dashboard
=================================================================
Author: Dashboard Team Member
This file is NEW - it does NOT modify any existing team files.
It reads: Candidate Profile.csv, Call log.csv, Executive Profile.csv, churn_prediction_model.pkl
"""

import re
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
import requests
from dotenv import load_dotenv
from datetime import datetime

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

# 1. INITIALIZE SUPABASE ROUTINES
@st.cache_resource
def init_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

@st.cache_resource
def init_supabase_service() -> Client:
    """Creates a service-role Supabase client that bypasses RLS for unrestricted data reads.
    Requires SUPABASE_SERVICE_KEY in .env — get it from:
    Supabase Dashboard -> Project Settings -> API -> service_role secret
    """
    try:
        url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
        service_key = os.environ.get("SUPABASE_SERVICE_KEY") or st.secrets.get("SUPABASE_SERVICE_KEY", "")
    except Exception:
        url = os.environ.get("SUPABASE_URL", "")
        service_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        # 3. Crash early with a helpful explanation if strings are missing
        if not url:
            raise ValueError("Initialization Failed: 'SUPABASE_URL' could not be resolved from secrets or environment.")
        if not service_key:
            raise ValueError(
                "Initialization Failed: 'SUPABASE_SERVICE_ROLE_KEY' could not be resolved from secrets or environment.")

        try:
            return create_client(url, service_key)
        except Exception as init_err:
            raise RuntimeError(f"Failed to establish Supabase client connection: {init_err}")
    return create_client(url, service_key)

supabase = init_supabase()
supabase_service = init_supabase_service()  # None if service key not configured


# 2. ADD RISK ANALYSIS HELPER
def identify_risk_factors(input_data: dict) -> dict:
    risk_factors = {}
    total = float(input_data.get('Total_Amount') or 0.0)
    paid = float(input_data.get('Paid_amount') or 0.0)
    paid_rate = float(input_data.get('Paid_Rate') or 0.0)

    if paid_rate < 0.4 or (total > 0 and (paid / total) < 0.4):
        risk_factors["low_payment_rate"] = True
    if total > 50000:
        risk_factors["high_total_amount"] = True
    if input_data.get('Test') == "No":
        risk_factors["no_test_completed"] = True
    if input_data.get('Followup Email') == "No":
        risk_factors["no_followup_engagement"] = True
    if int(input_data.get('Experience') or 0) == 0:
        risk_factors["zero_experience"] = True
    if input_data.get('Mode of Program Joined') == "Online":
        risk_factors["online_program"] = True
    if input_data.get('Feedback') and "poor" in str(input_data.get('Feedback')).lower():
        risk_factors["negative_feedback"] = True

    return risk_factors if risk_factors else {"general_risk": True}


def log_crm_call_interaction(
    email: str, duration_sec: int, agent_name: str, direction: str,
    owner_id: str, remark_cat: str = None,
    outcomes_list: list = None, summary_text: str = None, interest: str = "medium",
    followup_req: bool = False,
    next_followup_str: str = None, priority: str = None
    ):
    """Resolves transactional key bindings and writes communication history logs."""


    try:
        # A. Resolve mandatory candidate_id (UUID) via email tracking keys
        candidate_query = supabase_service.table("candidates").select("id").eq("email", email).limit(1).execute()
        if not candidate_query.data:
            raise ValueError(f"No candidate record found matching email: '{email}'")

        candidate_uuid = candidate_query.data[0]["id"]

        # B. Grab latest prediction reference string if available
        prediction_uuid = None
        pred_query = supabase_service.table("predictions").select("id").eq("email", email).order("predicted_at", desc=True).limit(1).execute()
        if pred_query.data:
            prediction_uuid = pred_query.data[0]["id"]

        # C. Text analytics sentiment calculation
        if summary_text:
            joy_words = ['interested', 'excited', 'yes', 'perfect', 'join', 'good', 'agree']
            sad_words = ['expensive', 'cancel', 'no', 'busy', 'unable', 'drop', 'bad']
            lower_text = summary_text.lower()
            pos = sum(1 for w in joy_words if w in lower_text)
            neg = sum(1 for w in sad_words if w in lower_text)
            sentiment_score = round((pos - neg) / (pos + neg), 2) if (pos + neg) > 0 else 0.0
        else:
            sentiment_score = 0.0

        # D. Assemble call interaction configuration block
        call_payload = {
            "candidate_id": candidate_uuid,
            "prediction_id": prediction_uuid,
            "owner_id": owner_id,                    # <-- FIX: Added mapping directly here
            "call_duration": int(duration_sec),
            "call_agent": agent_name,                # Passes the resolved label text
            "call_direction": direction.lower().strip(),
            "outcomes": outcomes_list,
            "remark_category": remark_cat if remark_cat else None,
            "transcript_summary": summary_text,
            "sentiment_score": float(sentiment_score),
            "interest_level": interest.lower().strip(),
            "followup_required": bool(followup_req),
            "next_followup_date": next_followup_str if followup_req else None,
            "followup_priority": priority.lower().strip() if (followup_req and priority) else None
        }

        supabase_service.table("calls").insert(call_payload).execute()
        return True, "Success"

    except ValueError as val_err:
        print(f"Validation Error: {val_err}")
        return False, str(val_err)
    except Exception as db_err:
        print(f"Call logging exception: {db_err}")
        return False, f"Database error: {db_err}"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def live_groq_pipeline(text, api_key):
    if not text.strip():
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    allowed_outcomes = [
        "not_interested", "unreachable", "joined_competitor",
        "financial_issue", "already_working", "looking_for_job",
        "decision_pending", "converted"
    ]

    prompt = f"""
    You are an expert recruitment call analyzer. Analyze this transcript content: "{text}"

    Perform the following tasks:
    1. Translate the text into clear English if it's in Malayalam.
    2. Write a 1-2 sentence Summary indicating if the candidate will join or not, extracting the primary reason for churn if they decline.
    3. Categorize the overall call into exactly ONE of these allowed database outcome tags: {allowed_outcomes}

    Return your output strictly in this JSON layout:
    {{
        "summary": "1-2 sentence final verdict summary goes here",
        "call_outcome": "one_of_the_allowed_tags_here"
    }}
    """

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": f"You are a precise data extractor. You must output strictly in JSON format. The 'call_outcome' key MUST match one of these tokens exactly: {allowed_outcomes}."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens": 400
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return json.loads(response.json()["choices"][0]["message"]["content"])
        else:
            st.error(f"Groq Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Network Error: {str(e)}")
        return None

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
        'Financial issues': ['financial issue', 'financial', 'payment', 'pay', 'fee', 'emi', 'installment', 'finance'],
        'Lack of interest': ['lack of interest', 'not interested', 'no interest', 'lost interest', 'not keen', 'disinterested', 'no longer interested'],
        'Joined another institution': ['joined another', 'joined competitor', 'joined other', 'admission elsewhere', 'admitted', 'migrated to', 'joined institute', 'joined company', 'enrolled elsewhere'],
        'Communication gaps': ['communication gaps', 'no response', 'no pickup', 'unreachable', 'voicemail', 'did not pick', 'not reachable', 'no answer', 'call dropped', 'busy', 'no contact', 'not responding'],
        'Already working': ['already working'],
        'Looking for job': ['looking for job/internship'],
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
    .stSelectbox label p, .stNumberInput label p, .stTextInput label p, .stDateInput label p, .stTextArea label p {
        color: #a78bfa !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .stSelectbox > div, .stNumberInput > div, .stTextInput > div, .stDateInput > div, .stTextArea > div {
        background: rgba(15, 12, 41, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Metric delta */
    [data-testid="stMetricDelta"] { font-size: 12px; }

    /* ---------------------------------------------------
       MOBILE RESPONSIVENESS & ADAPTIVE DESIGN
       --------------------------------------------------- */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .page-header {
            margin-bottom: 20px !important;
            padding-bottom: 16px !important;
        }
        .page-header h1 {
            font-size: 24px !important;
        }
        .page-header p {
            font-size: 13px !important;
        }
        .kpi-card {
            height: auto !important;
            padding: 16px 12px !important;
            margin-bottom: 12px !important;
        }
        .kpi-value {
            font-size: 28px !important;
            margin-bottom: 4px !important;
        }
        .kpi-title {
            font-size: 11px !important;
            margin-bottom: 6px !important;
        }
        .kpi-icon {
            font-size: 20px !important;
            margin-bottom: 8px !important;
        }
        .kpi-sub {
            font-size: 11px !important;
        }
        .section-header {
            margin: 24px 0 12px 0 !important;
        }
        .section-header h2 {
            font-size: 14px !important;
        }
        .prediction-box-churn, .prediction-box-safe {
            padding: 20px 16px !important;
        }
        .pred-label {
            font-size: 32px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px !important;
            flex-wrap: wrap !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 4px !important;
            font-size: 14px !important;
        }
        /* Make sure tables are scrollable on mobile */
        .stDataFrame {
            overflow-x: auto !important;
        }
        /* Ensure candidate card adjusts padding */
        .candidate-card {
            padding: 16px !important;
        }
        /* Adjust navigation elements on mobile */
        .nav-item {
            padding: 8px 12px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# (Replicates model.py logic — files untouched)
# ─────────────────────────────────────────────

INPUT_DIR = "./data/"
OUTPUT_DIR = "./output/"

#@st.cache_data(ttl=60)
#def load_data():
    # Load the processed dataset with inferred churn and reasons
#    df_path = os.path.join(OUTPUT_DIR, "enrolled_processed.csv")
#    notes_path = os.path.join(OUTPUT_DIR, "notes_processed.csv")
    
#    if os.path.exists(df_path):
#        df = pd.read_csv(df_path)
#    else:
#        df = pd.DataFrame()
        
#    if os.path.exists(notes_path):
#        notes = pd.read_csv(notes_path)
#    else:
#        notes = pd.DataFrame()
        
#    return df, notes

@st.cache_data(ttl=2)
def load_data():
    df_path = os.path.join(OUTPUT_DIR, "enrolled_processed.csv")
    notes_path = os.path.join(OUTPUT_DIR, "notes_processed.csv")

    # ── Layer 1: Attempt Absolute Admin Supabase Pull ──
    try:
        # 🌟 FORCE BEYOND RLS: Explicitly use the admin service-role key client.
        # If your keys are bound to 'supabase_service', use it directly to ignore security walls.
        if supabase_service is not None:
            read_client = supabase_service
        else:
            # If the service key variable name differs in your main initialization block,
            # force-assign it here or fallback visibly.
            read_client = supabase

        all_records = []
        chunk_size = 1000
        start_row = 0

        while True:
            # We enforce ordering to guarantee zero row-drifting over chunks
            response = read_client.table("candidates") \
                .select("*") \
                .order("id", desc=False) \
                .range(start_row, start_row + chunk_size - 1) \
                .execute()

            if response.data and len(response.data) > 0:
                all_records.extend(response.data)
                if len(response.data) < chunk_size:
                    break
                start_row += chunk_size
            else:
                break

        if len(all_records) > 0:
            df = pd.DataFrame(all_records)

            # (Keep your existing rename_mapping block exactly as it is...)
            rename_mapping = {
                "candidate_name": "Contact Name",
                "contact_id": "Contact Id",
                "email": "Email ID",
                "contact_phone": "Whatsapp Number",
                "city": "City",
                "mailing_state": "Mailing State",
                "mailing_country": "Mailing Country",
                "course": "Course",
                "stream": "Stream",
                "track_interested": "Track Interested",
                "batch_assigned": "Batch Assigned",
                "program_mode": "Mode of Program Joined",
                "program_location": "Program Location",
                "induction_session": "Induction session",
                "background_override": "final_inferred_reason",
                "csv_contact_owner": "Contact Owner",
                "Payment_Date": "Payment Date",
                "Payment_mode": "Payment_mode",
                "Source of Lead": "Source of lead",
                "Feedback": "Feedback",
                "Invoice": "Invoice",
                "Experience": "Experience",
                "Test": "Test",
                "Followup Email": "Followup Email",
                "gender": "Gender",
                "education": "Education"
            }
            existing_renames = {k: v for k, v in rename_mapping.items() if k in df.columns}
            if existing_renames:
                df = df.rename(columns=existing_renames)

            notes = pd.read_csv(notes_path) if os.path.exists(notes_path) else pd.DataFrame()
            return df, notes, "database"

    except Exception as db_err:
        st.error(f"Database extraction pipeline interruption details: {db_err}")

    # ── FALLBACK LAYER: Load Local CSV Files ──
    df = pd.read_csv(df_path) if os.path.exists(df_path) else pd.DataFrame()
    notes = pd.read_csv(notes_path) if os.path.exists(notes_path) else pd.DataFrame()

    return df, notes, "csv"


@st.cache_data
def load_churn_reasons():
    # Deprecated/Not needed anymore, returning same df for compatibility
    return None, None


    # Data is already preprocessed by churn_data.py.
    # We just ensure certain columns exist to avoid KeyError in UI


@st.cache_data
def preprocess(df, notes):
    """
    Cleans structural schemas and renames 'background_override' directly to
    'final_inferred_reason' while handling data types and filling nulls with 'N/A'.
    """
    if df.empty:
        return df, notes

    # ── 1. PRE-CLEANING & NUMERIC STANDARD CONVERSIONS ──
    df['Experience'] = pd.to_numeric(df['Experience'], errors='coerce').fillna(0.0)
    df['Semester'] = pd.to_numeric(df['Semester'], errors='coerce').fillna(0).astype(int)
    df['Year of Graduation'] = pd.to_numeric(df['Year of Graduation'], errors='coerce').fillna(0).astype(int)
    if 'Paid_amount' in df.columns:
        df['Paid_amount'] = pd.to_numeric(df['Paid_amount'], errors='coerce').fillna(0.0)
    if 'Total_Amount' in df.columns:
        df['Total_Amount'] = pd.to_numeric(df['Total_Amount'], errors='coerce').fillna(0.0)

    df['Course'] = df['Course'].fillna('Unknown')
    df['Invoice'] = df['Invoice'].fillna('No')

    # ── 🌟 TROUBLESHOOTING FIX: RENAME, TYPECAST & FILL NULL VALUES ──
    df['final_inferred_reason'] = df['final_inferred_reason'].fillna('N/A')
        # 2. Rename directly to 'final_inferred_reason' for your charts/analysis pages
    #df = df.rename(columns={'background_override': 'final_inferred_reason'})


    # Ensure it is explicitly typed as object/string in the pandas schema
    #df['final_inferred_reason'] = df['final_inferred_reason'].astype(str)

    # ── 2. DYNAMIC 'role' DERIVATION ──
    def assign_role(row):
        if row['Experience'] > 0:
            return 'Professional'
        elif row['Semester'] > 0:
            return 'Student'
        elif row['Year of Graduation'] > 0 and row['Experience'] == 0:
            return 'Idle or Career Gap'
        return 'Unknown'

    df['role'] = df.apply(assign_role, axis=1)

    # ── 3. DYNAMIC 'background' DERIVATION FROM EDUCATION ──
    def assign_background(course):
        if pd.isna(course) or str(course).strip().upper() in ['NOT MENTIONED', 'UNSPECIFIED', 'UNKNOWN', '']:
            return 'UNKNOWN'

        tech_keywords = [
            'BTECH', 'BE', 'MTECH', 'BCA', 'MCA', 'B VOC-IT', 'BSC-CS', 'MSC-CS', 'CSE', 'CS', 'IT', 'DA', 'DS', 'BDA',
            'MSCIT', 'BSCIT', 'MTECHIT', 'MSC-CS-DA', 'BTECH-IT', 'BSC-IT', 'DIPLOMA-IT'
        ]
        non_tech_keywords = [
            'BCOM', 'MCOM', 'BA', 'MBA', 'MA', 'B VOC', 'BSC', 'MSC', 'ENG LIT', 'PLUS TWO', 'DIPLOMA', 'PG',
            'BSC-NON-IT', 'DIPLOMA-NON-IT', 'GRADUATED', 'OTHERS'
        ]

        course_upper = str(course).upper()
        for keyword in tech_keywords:
            if keyword in course_upper:
                return 'Tech'
        for keyword in non_tech_keywords:
            if keyword in course_upper:
                return 'Non-Tech'

        if 'TECH' in course_upper or 'SCIENCE' in course_upper or 'ENGINEERING' in course_upper:
            return 'Tech'
        elif 'ARTS' in course_upper or 'COMMERCE' in course_upper or 'HUMANITIES' in course_upper:
            return 'Non-Tech'

        return 'UNKNOWN'

    edu_col = 'Education' if 'Education' in df.columns else 'education'
    if edu_col in df.columns:
        df['background'] = df[edu_col].apply(assign_background)
    else:
        df['background'] = 'UNKNOWN'

    # ── 4. DYNAMIC TARGET 'Status' MATRIX DERIVATION ──
    def assign_status(row):
        inv_val = str(row['Invoice']).strip().lower()
        reason_val = str(row['final_inferred_reason']).strip()

        is_invoice_yes = inv_val in ['yes', 'paid']
        is_reason_na = reason_val in ['N/A', 'n/a', '', 'None', 'nan', 'none']

        if is_invoice_yes and is_reason_na:
            return 'Joined'
        elif is_invoice_yes and not is_reason_na:
            return 'Churned'
        elif not is_invoice_yes and not is_reason_na:
            return 'Not joined'
        elif not is_invoice_yes and is_reason_na:
            return 'Yet to pay'

        return 'Yet to pay'

    df['Status'] = df.apply(assign_status, axis=1)

    return df, notes


@st.cache_resource(ttl=60)
def load_model():
    try:
        with open(os.path.join(OUTPUT_DIR, 'prediction_model.pkl'), 'rb') as f:
            model_data = pickle.load(f)

            # Add missing keys with defaults (same as before)
            if 'model_name' not in model_data and 'model' in model_data:
                model_data['model_name'] = model_data['model'].__class__.__name__
            if 'model_display_name' not in model_data:
                model_data['model_display_name'] = model_data.get('model_name', 'Unknown Model')
            if 'balance_method' not in model_data:
                model_data['balance_method'] = 'none'

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
                Admin & Sales Portal
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 42px; font-weight: 700; letter-spacing: 0.5px; line-height: 1.1;">
                <span style="color: #f8fafc;">Churn</span><span style="background: -webkit-linear-gradient(45deg, #a78bfa, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Sense</span>
                <span style="font-family: 'Inter', sans-serif; font-size: 15px; font-weight: 900; letter-spacing: 1px; color: #38bdf8; vertical-align: top; margin-left: 2px;">AI</span>
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #94a3b8; margin-top: 10px; text-transform: uppercase; letter-spacing: 3.5px; font-weight: 500;">
                Admin & Sales Portal
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
                <h1 style="font-family: 'Playfair Display', serif; font-size: 42px; margin: 0; color: #f8fafc; font-weight: 700;">System Login</h1>
                <p style="color: #94a3b8; font-size: 14px; margin-top: 8px;">Access your secure workspace</p>
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
                            res = supabase.auth.sign_in_with_password(
                                {"email": login_email, "password": login_password})
                            st.session_state.logged_in = True
                            st.session_state.user_email = res.user.email

                            # 🌟 Extract role from user metadata (defaults to Salesperson if not defined)
                            user_metadata = res.user.user_metadata if res.user.user_metadata else {}
                            st.session_state.user_role = user_metadata.get("role", "Salesperson")

                            # Set appropriate default landing page depending on the role
                            if st.session_state.user_role == "Admin":
                                st.session_state.current_page = "Overview"
                            else:
                                st.session_state.current_page = "Smart Agent Workspace"

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
                <p style="color: #94a3b8; font-size: 14px; margin-top: 8px;">Register for enterprise access</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("register_form", border=False):
                reg_email = st.text_input("Email Address", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_password")

                # 🌟 Added Role selection drop-down directly into Registration Form
                reg_role = st.selectbox("Select Account Tier", ["Salesperson", "Admin"], index=0, key="reg_role")

                st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
                submitted_reg = st.form_submit_button("Sign Up", type="primary", use_container_width=True)
                if submitted_reg:
                    if reg_email and reg_password:
                        try:
                            # 🌟 Pass the selected role metadata directly into Supabase User Sign Up configuration
                            res = supabase.auth.sign_up({
                                "email": reg_email,
                                "password": reg_password,
                                "options": {
                                    "data": {
                                        "role": reg_role
                                    }
                                }
                            })
                            st.success(
                                f"Registration successful as {reg_role}! You can now log in using the Login tab.")
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
                <div style="margin-top:5px;"><span class="badge-low" style="font-size:10px;">Role: {st.session_state.get('user_role', 'Salesperson')}</span></div>
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
                <div style="margin-top:5px;"><span class="badge-low" style="font-size:10px;">Role: {st.session_state.get('user_role', 'Salesperson')}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # 🌟 Define access configurations depending on assigned roles
        user_role = st.session_state.get("user_role", "Salesperson")

        if user_role == "Admin":
            pages = [
                ("Overview", ":material/dashboard:"),
                ("Candidate Explorer", ":material/search:"),
                ("Salesperson Analytics", ":material/analytics:"),  # 🌟 ADDED THIS
                ("Smart Agent Workspace", ":material/assignment_turned_in:"),
                ("Add New Candidate", ":material/person_add:"),
                ("CRM Notes Analysis", ":material/call:"),
                ("Invoice Analysis", ":material/payments:"),
                ("Live Predictor", ":material/online_prediction:"),
                ("Model Performance", ":material/insights:")
            ]
        else:
            # Salesperson gets visibility access ONLY to specified scope paths
            pages = [
                ("Smart Agent Workspace", ":material/assignment_turned_in:"),
                ("Add New Candidate", ":material/person_add:"),
                ("Live Predictor", ":material/online_prediction:")
            ]

        # Double check that the current fallback page exists inside the user's available structural options
        if "current_page" not in st.session_state or st.session_state.current_page not in [p[0] for p in pages]:
            st.session_state.current_page = pages[0][0]

        for p_name, p_icon in pages:
            if st.button(p_name, icon=p_icon, use_container_width=True):
                st.session_state.current_page = p_name
                st.session_state.show_profile = False
                st.rerun()

        # 🌟 DYNAMICALLY HIGHLIGHT BUTTON BASED ON VISIBLE INDEX POSITION
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
        st.markdown(
            "<p style='font-size:10px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;margin:0 0 8px 0;'>Data Sources</p>",
            unsafe_allow_html=True)

        _src = [
            ("<i class='fa-solid fa-users'></i>", "Enrolled & Registered", ".xlsx &nbsp;·&nbsp; 1084 rows"),
            ("<i class='fa-solid fa-clipboard-list'></i>", "CRM All Contacts", ".xlsx &nbsp;·&nbsp; 1084 rows"),
            ("<i class='fa-solid fa-file-invoice'></i>", "Notes Processed", ".csv &nbsp;·&nbsp; 30K+ rows"),
            ("<i class='fa-solid fa-robot'></i>", "Churn Model", ".pkl &nbsp;·&nbsp; Saved Model"),
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
    <b style="color:#34d399;">Interactive Console</b> &mdash; data additions authorized
  </span>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        if st.button("Log Out", icon=":material/logout:", type="secondary", use_container_width=True):
            st.session_state.logged_in = False
            if "access_token" in st.session_state:
                del st.session_state["access_token"]
            if "refresh_token" in st.session_state:
                del st.session_state["refresh_token"]
            if "user_role" in st.session_state:
                del st.session_state["user_role"]
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
    churned     = (df['Status'] == 'Churned').sum()
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
    if 'final_inferred_reason' in df.columns:
        try:
            unique_churn = df[df['Status'] == 'Churned'].drop_duplicates(subset=['Contact Id'])
            mapped_reasons = unique_churn['final_inferred_reason'].dropna().astype(str).apply(normalize_reason_label)
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
        src = df.groupby(['Source of lead', 'Status']).size().reset_index(name='Count')
        src['Status'] = src['Status'].map({'Churned': 'Churned'})
        fig2 = px.bar(src, x='Source of lead', y='Count', color='Status',
                      color_discrete_map={'Churned': COLOR_CHURN},
                      barmode='group', text='Count')
        fig2.update_traces(textfont_size=11, textposition='outside')
        fig2.update_layout(**theme(height=310, showlegend=True))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Course + Background + Mode ─────────
    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown('<div class="section-header"><h2>By Course</h2></div>', unsafe_allow_html=True)
        course_churn = df[df['Status'] == 'Churned']['Course'].value_counts().reset_index()
        course_churn.columns = ['Course', 'Churned']
        fig3 = px.bar(course_churn, x='Churned', y='Course', orientation='h',
                      color='Churned', color_continuous_scale=['#4c1d95','#f87171'])
        fig3.update_layout(**theme(height=280, showlegend=False,
                                   coloraxis_showscale=False))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header"><h2>Background Split</h2></div>', unsafe_allow_html=True)
        bg = df.groupby(['background', 'Status']).size().reset_index(name='Count')
        bg['Status'] = bg['Status'].map({'Churned': 'Churned'})
        fig4 = px.pie(bg, names='background', values='Count',
                      color='background', hole=0.5,
                      color_discrete_sequence=PALETTE)
        fig4.update_layout(**theme(height=280, showlegend=True))
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.markdown('<div class="section-header"><h2>Training Mode</h2></div>', unsafe_allow_html=True)
        mode_data = df.groupby(['Mode of Program Joined', 'Status']).size().reset_index(name='Count')
        mode_data['Status'] = mode_data['Status'].map({'Churned': 'Churned'})
        fig5 = px.bar(mode_data, x='Mode of Program Joined', y='Count', color='Status',
                      color_discrete_map={'Churned': COLOR_CHURN},
                      barmode='stack')
        fig5.update_layout(**theme(height=280))
        st.plotly_chart(fig5, use_container_width=True)

    # ── Row 3: Role + Inferred Reason ───────────────
    col6, col7 = st.columns(2)
    with col6:
        st.markdown('<div class="section-header"><h2>Candidate Role vs Churn</h2></div>', unsafe_allow_html=True)
        ind = df.groupby(['role', 'Status']).size().reset_index(name='Count')
        ind['Status'] = ind['Status'].map({'Churned': 'Churned'})
        fig6 = px.bar(ind, x='role', y='Count', color='Status',
                      color_discrete_map={'Churned': COLOR_CHURN},
                      barmode='group', text='Count')
        fig6.update_traces(textfont_size=11, textposition='outside')
        fig6.update_layout(**theme(height=290))
        st.plotly_chart(fig6, use_container_width=True)

    with col7:
        st.markdown('<div class="section-header"><h2>Induction Session Attended vs Churn</h2></div>', unsafe_allow_html=True)
        fb = df.groupby(['Induction session', 'Status']).size().reset_index(name='Count')
        # Limit to top 10 reasons to avoid chart clutter
        fb = fb.sort_values('Count', ascending=False).head(20)
        fb['Status'] = fb['Status'].map({'Joined': 'Active', 'Churned': 'Churned'})
        fig7 = px.bar(fb, x='Induction session', y='Count', color='Status',
                      color_discrete_map={'Avtive': COLOR_ACTIVE,'Churned': COLOR_CHURN},
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

    if sel_churn == "Churned": fdf = fdf[fdf['Status'] == 'Churned']
    elif sel_churn == "Active": fdf = fdf[fdf['Status'] == 'Joined']
    if sel_source != "All": fdf = fdf[fdf['Source of lead'] == sel_source]
    if sel_course != "All": fdf = fdf[fdf['Course'] == sel_course]
    if sel_mode   != "All": fdf = fdf[fdf['Mode of Program Joined']   == sel_mode]
    if sel_bg     != "All": fdf = fdf[fdf['background'] == sel_bg]

    st.markdown(f"<p style='color:#64748b; font-size:13px;'>Showing <b style='color:#a78bfa'>{len(fdf)}</b> candidates</p>", unsafe_allow_html=True)

    # ── Table ─────────────────────────────────────
    display_cols = ['Contact Id', 'Contact Name', 'Education', 'Induction session', 'Feedback',  'Source of lead', 'Course', 'Mode of Program Joined',
                    'background', 'role', 'Status', 'Invoice', 'final_inferred_reason']
    display_cols = [c for c in display_cols if c in fdf.columns]

    tbl = fdf[display_cols].copy()
    tbl['Status'] = tbl['Status'].map({'Joined': 'Active', 'Churned': 'Churned'})

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
            "Status":                 st.column_config.TextColumn("Status"),
            "Invoice":                st.column_config.TextColumn("Invoice Status"),
            "final_inferred_reason":  st.column_config.TextColumn("AI Insight"),
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

    churn_label = "CHURNED" if row.get('Status', 'Churned') == 'Churned' else "ACTIVE"
    churn_color = "#f87171" if row.get('Status', 'Churned') == 'Churned' else "#34d399"

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
    if 'final_inferred_reason' in row.index and pd.notna(row['final_inferred_reason']) and row['final_inferred_reason'] != '':
        reason_data = row['final_inferred_reason']
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
# Page 3: Sales person analysis
# ─────────────────────────────────────────────
def page_salesperson_stats(candidates_df, supabase):
    st.markdown(
        '<div class="page-header"><h1><i class="fa-solid fa-chart-line"></i> Salesperson Analytics</h1><p>Admin Control Dashboard: Live tracking across profiles, system predictions, and risk vectors.</p></div>',
        unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # 1. LIVE DATA FETCHING & MAPPING PIPELINE
    # ─────────────────────────────────────────────
    @st.cache_data(ttl=60)  # Cache for 1 minute to keep app fast
    def fetch_relational_metrics():
        try:
            # A. Fetch mappings: legacy_label (name) <-> salesperson_email
            map_res = supabase.table("salesperson_mappings").select("legacy_label, salesperson_email").execute()
            map_df = pd.DataFrame(map_res.data) if map_res.data else pd.DataFrame(
                columns=["legacy_label", "salesperson_email"])

            # B. Fetch prediction logs
            pred_res = supabase.table("predictions").select("predicted_by, risk_level").execute()
            preds_raw = pd.DataFrame(pred_res.data) if pred_res.data else pd.DataFrame(
                columns=["predicted_by", "risk_level"])

            return map_df, preds_raw
        except Exception as e:
            st.error(f"Error fetching structural relational data maps: {e}")
            return pd.DataFrame(), pd.DataFrame()

    with st.spinner("Compiling cross-table salesperson relationships..."):
        map_df, preds_df = fetch_relational_metrics()

    if map_df.empty:
        st.warning("Salesperson mapping matrix table is empty. Relational graphics cannot resolve names correctly.")
        return

    # Clean text formatting to ensure join matches work flawlessly
    map_df['legacy_label'] = map_df['legacy_label'].fillna("").str.strip().str.lower()
    map_df['salesperson_email'] = map_df['salesperson_email'].fillna("").str.strip().str.lower()

    # Build dictionary maps for fast bidirectional looking up
    name_to_email = dict(zip(map_df['legacy_label'], map_df['salesperson_email']))
    email_to_name = dict(zip(map_df['salesperson_email'], map_df['legacy_label']))

    owner_col = 'Contact Owner'
    status_col = 'Status'

    # Convert candidates owner names to lowercase for robust dictionary key matching
    candidates_df['cleaned_owner'] = candidates_df[owner_col].fillna("unassigned").str.strip().str.lower()

    # Map raw candidate owner strings to their official corporate Email IDs
    candidates_df['mapped_email'] = candidates_df['cleaned_owner'].map(name_to_email).fillna(
        candidates_df['cleaned_owner'])

    # Standardize predictions dataset text rows
    if not preds_df.empty:
        preds_df['predicted_by'] = preds_df['predicted_by'].fillna("unknown").str.strip().str.lower()
        preds_df['risk_level'] = preds_df['risk_level'].astype(str).str.strip().str.capitalize()
        # Create a display label for graphs so admins see human-readable names instead of raw email strings
        preds_df['display_name'] = preds_df['predicted_by'].map(email_to_name).fillna(preds_df['predicted_by'])
    else:
        preds_df['display_name'] = pd.Series(dtype='str')

    # ─────────────────────────────────────────────
    # 2. METRIC MATRIX GENERATION
    # ─────────────────────────────────────────────
    # A. Registration metrics
    total_reg_counts = candidates_df['mapped_email'].value_counts()
    best_sp_email = total_reg_counts.idxmax() if not total_reg_counts.empty else "N/A"
    best_sp_display = email_to_name.get(best_sp_email, best_sp_email)
    best_sp_count = total_reg_counts.max() if not total_reg_counts.empty else 0

    # B. Churn processing using mapped Email context references
    churn_mask = candidates_df[status_col].astype(str).str.lower().str.contains('churn', na=False)
    churn_counts_by_sp = candidates_df[churn_mask]['mapped_email'].value_counts()
    churn_rates = (churn_counts_by_sp / total_reg_counts).fillna(0)
    highest_churn_email = churn_rates.idxmax() if not churn_rates.empty and churn_rates.max() > 0 else "None"
    highest_churn_display = email_to_name.get(highest_churn_email, highest_churn_email)
    highest_churn_val = f"{churn_rates.max() * 100:.1f}%" if highest_churn_email != "None" else "0%"

    # C. Prediction matrix calculations
    total_preds_run = len(preds_df)
    high_risk_total = preds_df[preds_df['risk_level'] == 'High'].shape[0] if not preds_df.empty else 0

    # Render KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card kpi-blue">
            <div class="kpi-icon"><i class="fa-solid fa-trophy"></i></div>
            <div class="kpi-title">Top Recruiter</div>
            <div class="kpi-value" style="font-size:18px; overflow:hidden; text-overflow:ellipsis;">{best_sp_display}</div>
            <div class="kpi-sub">{best_sp_count} Candidates Registered</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card kpi-red">
            <div class="kpi-icon"><i class="fa-solid fa-user-slash"></i></div>
            <div class="kpi-title">Highest Churn Risk</div>
            <div class="kpi-value" style="font-size:18px; overflow:hidden; text-overflow:ellipsis;">{highest_churn_display}</div>
            <div class="kpi-sub">Loss Ratio: {highest_churn_val}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card kpi-green">
            <div class="kpi-icon"><i class="fa-solid fa-calculator"></i></div>
            <div class="kpi-title">Total Inferences Run</div>
            <div class="kpi-value">{total_preds_run}</div>
            <div class="kpi-sub">Across all connected platform agents</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card kpi-amber">
            <div class="kpi-icon"><i class="fa-solid fa-triangle-exclamation"></i></div>
            <div class="kpi-title">High Risk Warnings</div>
            <div class="kpi-value">{high_risk_total}</div>
            <div class="kpi-sub">Flagged anomalies identified</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # 3. GRAPHICAL VISUALIZATIONS
    # ─────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header"><h2>Candidates Registered per Salesperson</h2></div>',
                    unsafe_allow_html=True)
        # For readability on graphs, map the emails back to friendly local names
        graph_reg_data = total_reg_counts.reset_index()
        graph_reg_data.columns = ['Salesperson Email', 'Count']
        graph_reg_data['Salesperson Name'] = graph_reg_data['Salesperson Email'].map(email_to_name).fillna(
            graph_reg_data['Salesperson Email'])

        fig_reg = px.bar(graph_reg_data, x='Count', y='Salesperson Name', orientation='h',
                         template='plotly_dark', color='Count',
                         color_continuous_scale=['#a78bfa', '#38bdf8'])
        fig_reg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
                              yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_reg, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header"><h2>Salesperson vs Churn Status</h2></div>', unsafe_allow_html=True)
        candidates_df['Salesperson Name'] = candidates_df['mapped_email'].map(email_to_name).fillna(
            candidates_df['mapped_email'])
        churn_mix = candidates_df.groupby(['Salesperson Name', status_col]).size().reset_index(name='Count')

        fig_churn = px.bar(churn_mix, x='Salesperson Name', y='Count', color=status_col,
                           template='plotly_dark', barmode='stack',
                           color_discrete_sequence=['#f87171', '#34d399', '#60a5fa'])
        fig_churn.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                xaxis_title="Salesperson Name")
        st.plotly_chart(fig_churn, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-header"><h2>Predictions Executed by Account</h2></div>',
                    unsafe_allow_html=True)
        if not preds_df.empty:
            pred_counts = preds_df['display_name'].value_counts().reset_index()
            pred_counts.columns = ['Salesperson Name', 'Total Runs']

            fig_pred = px.pie(pred_counts, values='Total Runs', names='Salesperson Name', template='plotly_dark',
                              hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_pred.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info("No active prediction records discovered.")

    with c4:
        st.markdown('<div class="section-header"><h2>High Risk Analysis Distribution</h2></div>',
                    unsafe_allow_html=True)
        if not preds_df.empty and 'risk_level' in preds_df.columns:
            risk_mix = preds_df.groupby(['display_name', 'risk_level']).size().reset_index(name='Count')

            fig_risk = px.bar(risk_mix, x='Count', y='display_name', color='risk_level', orientation='h',
                              template='plotly_dark', barmode='group',
                              color_discrete_map={'High': '#f87171', 'Medium': '#fbbf24', 'Low': '#34d399'})
            fig_risk.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   yaxis_title="Salesperson Name")
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("Risk metrics unavailable or not configured inside predictions.")

# ─────────────────────────────────────────────
## PAGE 4 — Call log
# ─────────────────────────────────────────────

def render_agent_workspace_and_logger(supabase, active_owner_uuid):
    """
    Renders the custom styled Smart Agent Workspace with KPI matrix summary
    cards, priority followup task queues, and an integrated interaction logger.
    """
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-assignment-turned-in"></i> Smart Agent Workspace</h1>
        <p>Manage prioritized task pipelines, track active customer risk metrics, and log communications.</p>
    </div>
    """, unsafe_allow_html=True)

    current_user_email = st.session_state.get("user_email")
    # ── STAGE 1: FETCH THE USER ID FROM SUPABASE AUTH ────────────────
    active_owner_uuid = None
    try:
        # Fetch users from Supabase Auth admin panel
        auth_response = supabase_service.auth.admin.list_users()

        # The SDK returns an object where the user list is attached to `.users`
        # or can be unpacked directly if handled as a raw data wrapper
        users_list = getattr(auth_response, 'users', auth_response)

        if users_list:
            # Loop through the user list to find the exact email match
            for user in users_list:
                if hasattr(user,
                           'email') and user.email.lower().strip() == current_user_email.lower().strip():
                    active_owner_uuid = user.id
                    break

    except Exception as e:
        print(f"Error querying Supabase Auth Admin table: {e}")

    # ── 1. CORE PERFORMANCE OVERVIEW MATRIX (KPI CARDS) ───────────────────
    try:
        rpc_stats = supabase.rpc("get_dashboard_stats", {"p_owner_id": active_owner_uuid}).execute()
        if rpc_stats.data:
            s = rpc_stats.data[0]

            # Safely extract with fallbacks to avoid NoneType errors
            total_leads = s.get("total_candidates") or 0
            high_risk = s.get("high_risk_count") or 0
            urgent_reminders = s.get("urgent_followups") or 0

            # FIX: Provide a default fallback string or float before calling float()
            avg_risk = float(s.get("avg_churn_probability") or 0.0)

            kpi_html = f"""
            <div style="display:flex; gap:16px; margin-bottom:24px; flex-wrap: wrap;">
                <div style="flex:1; min-width:180px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; text-align:center;">
                    <div style="font-size:32px; font-weight:800; color:#38bdf8;">{total_leads}</div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:4px; font-weight:500; text-transform:uppercase; letter-spacing:1px;">Leads Assigned</div>
                </div>
                <div style="flex:1; min-width:180px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; text-align:center;">
                    <div style="font-size:32px; font-weight:800; color:#ef4444;">{high_risk}</div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:4px; font-weight:500; text-transform:uppercase; letter-spacing:1px;">High Risk Churns</div>
                </div>
                <div style="flex:1; min-width:180px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; text-align:center;">
                    <div style="font-size:32px; font-weight:800; color:#fbbf24;">{urgent_reminders}</div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:4px; font-weight:500; text-transform:uppercase; letter-spacing:1px;">Urgent Actions</div>
                </div>
                <div style="flex:1; min-width:180px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; text-align:center;">
                    <div style="font-size:32px; font-weight:800; color:#a78bfa;">{avg_risk:.1%}</div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:4px; font-weight:500; text-transform:uppercase; letter-spacing:1px;">Avg Risk Baseline</div>
                </div>
            </div>
            """
            st.markdown(kpi_html, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"KPI compilation error: {e}")

    # ── 2. SMART TASK REMINDERS QUEUE ─────────────────────────────────────
    st.markdown('<div class="section-header"><h2> Prioritized Action Items (7 Days)</h2></div>',
                unsafe_allow_html=True)
    try:
        reminders = supabase.rpc("get_followup_reminders", {"p_owner_id": active_owner_uuid}).execute()
        if reminders.data:
            df_reminders = pd.DataFrame(reminders.data)

            # Formatted column layout summary
            st.dataframe(
                df_reminders[[
                    "next_followup_date", "followup_priority", "candidate_name",
                    "email", "course", "churn_probability", "interest_level"
                ]].reset_index(drop=True),
                use_container_width=True,
                height=240,
                column_config={
                    "next_followup_date": st.column_config.TextColumn("Due Date", width="small"),
                    "followup_priority": st.column_config.TextColumn("Priority"),
                    "candidate_name": st.column_config.TextColumn("Student Name"),
                    "email": st.column_config.TextColumn("Email Address"),
                    "course": st.column_config.TextColumn("Course"),
                    "churn_probability": st.column_config.NumberColumn("AI Risk", format="%.1%"),
                    "interest_level": st.column_config.TextColumn("Interest"),
                }
            )
        else:
            st.success(" All clear! There are no pending follow-up tasks on your calendar for this week.")
    except Exception as e:
        st.info("No active follow-up entries resolved.")

    # ── 3. INTERACTIVE TOUCHPOINT LOGGING FORM ─────────────────────────────
    st.markdown('<div class="section-header"><h2> Log Live Communication Interaction</h2></div>',
                unsafe_allow_html=True)

    with st.expander("Open Communication Ingestion Terminal", expanded=True):
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                c_email = st.text_input("Candidate Target Email Reference", placeholder="student@example.com")
                duration_sec = st.number_input("Call Duration Metrics (Seconds)", min_value=0, value=60, step=10)
                interest = st.selectbox("Inferred Interest Level", ["High", "Medium", "Low"], index=1)

            with col2:
                direction = st.selectbox("Interaction Direction", ["outbound", "inbound"])
                remark_cat = st.selectbox("Call Remark",
                                          ['positive', 'negative', 'neutral', 'follow_up_required', 'resolved', 'callback_requested', 'not_interested', 'pricing_concern', 'time_constraint', 'need_more_info'])

            st.markdown('<div style="margin-top: 15px; margin-bottom: 5px; font-size:14px; font-weight:600; color:#cbd5e1; border-bottom: 1px solid #334155; padding-bottom: 5px;"><i class="fa-solid fa-calendar-check" style="margin-right:8px; color:#38bdf8;"></i> Follow-up Action Plan</div>', unsafe_allow_html=True)
            f_req = st.toggle("Enable Future Follow-up", value=False)
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                f_date = st.date_input("Followup Date", min_value=datetime.today(), disabled=not f_req)
            with col_f2:
                f_pri = st.selectbox("Followup Priority", ["low", "medium", "high", "urgent"], index=1, disabled=not f_req)

            if "current_summary" not in st.session_state:
                st.session_state["current_summary"] = ""
            # This will quietly hold onto the Enum string behind the scenes
            if "hidden_outcome" not in st.session_state:
                st.session_state["hidden_outcome"] = ""
            if "previous_text" not in st.session_state:
                st.session_state["previous_text"] = ""

            # 1. User inputs transcript
            transcript_text = st.text_area(
                "Full Audio Call Transcription",
                placeholder="Paste conversation transcription details here...",
                height=200
            )

            # Watch for Paste Event (Auto-generates summary and outcome)
            if transcript_text.strip() and transcript_text != st.session_state["previous_text"]:
                if not GROQ_API_KEY:
                    st.error("Missing GROQ_API_KEY in your local .env configuration.")
                else:
                    with st.spinner("Processing Malayalam text..."):
                        result = live_groq_pipeline(transcript_text, GROQ_API_KEY)
                        if result:
                            st.session_state["current_summary"] = result.get("summary", "")
                            st.session_state["hidden_outcome"] = result.get("call_outcome", "")
                            st.session_state["previous_text"] = transcript_text
                            st.rerun()

            # Reset state if text box is cleared
            if not transcript_text.strip() and st.session_state["current_summary"]:
                st.session_state["current_summary"] = ""
                st.session_state["hidden_outcome"] = ""
                st.session_state["previous_text"] = ""
                st.rerun()

            # 2. Display Generated Summary (The user can read/edit this if necessary)
            remarks_text = st.text_area(
                "Transcription Summary",
                value=st.session_state["current_summary"],
                placeholder="Summary will auto-generate here...",
                height=100
            )


            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            if st.button("Commit Log Ingestion", type="primary", use_container_width=True):
                if not c_email:
                    st.error("Please specify a target email identity mapping reference.")
                else:
                    f_date_str = f_date.strftime("%Y-%m-%d") if f_req else None

                    # ── STAGE 1: FETCH THE USER ID FROM SUPABASE AUTH ────────────────
                    active_owner_uuid = None
                    try:
                        # Fetch users from Supabase Auth admin panel
                        auth_response = supabase_service.auth.admin.list_users()

                        # The SDK returns an object where the user list is attached to `.users`
                        # or can be unpacked directly if handled as a raw data wrapper
                        users_list = getattr(auth_response, 'users', auth_response)

                        if users_list:
                            # Loop through the user list to find the exact email match
                            for user in users_list:
                                if hasattr(user,
                                           'email') and user.email.lower().strip() == current_user_email.lower().strip():
                                    active_owner_uuid = user.id
                                    break

                    except Exception as e:
                        print(f"Error querying Supabase Auth Admin table: {e}")

                    # ── STAGE 2: FETCH THE LEGACY LABEL FROM SALESPERSON MAPPING ─────
                    legacy_agent_label = "Unknown Agent"
                    try:
                        agent_query = supabase.table("salesperson_mappings") \
                            .select("legacy_label") \
                            .eq("salesperson_email", current_user_email.strip()) \
                            .limit(1).execute()

                        if agent_query.data and agent_query.data[0].get("legacy_label"):
                            legacy_agent_label = agent_query.data[0]["legacy_label"]
                    except Exception as e:
                        print(f"Error looking up legacy mapping table: {e}")

                    # ── STAGE 3: VALIDATION AND EXECUTION ────────────────────────────
                    if not active_owner_uuid:
                        st.error(
                            f" Ingestion Blocked: Found your session email ('{current_user_email}'), but it does not map to any registered account in Supabase Auth.")
                    else:
                        # Execute log insertion using the freshly resolved UUID
                        success, message = log_crm_call_interaction(
                            email=c_email.strip(),
                            duration_sec=int(duration_sec),
                            agent_name=legacy_agent_label,
                            owner_id=active_owner_uuid,  # <-- Pass the real Auth UUID here!
                            direction=direction,
                            outcomes_list=[st.session_state["hidden_outcome"]],
                            remark_cat=remark_cat,
                            summary_text=remarks_text[:200] if remarks_text else "No summary provided.",
                            interest=interest,
                            followup_req=f_req,
                            next_followup_str=f_date_str,
                            priority=f_pri
                        )

                        if success:
                            st.success("Interaction touchpoint committed directly to transactional registries.")
                            st.rerun()
                        else:
                            st.error(f"Transaction Aborted: {message}")

# ─────────────────────────────────────────────
# Page 5: Add candidate
# ─────────────────────────────────────────────
def render_candidate_entry_form(df, notes):
    # ── Identical Page Header Layout ─────────────────────────────
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-user-plus"></i> Add New Candidate</h1>
        <p>Manually create and synchronize a fresh candidate profile into the CRM database.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pure dynamic extractor: No hardcoded fallbacks allowed
    def get_exact_dataset_options(col_name):
        if col_name in df.columns:
            unique_vals = df[col_name].dropna().unique()
            cleaned = sorted([str(x).strip() for x in unique_vals if str(x).strip() != ''])
            if cleaned:
                return cleaned
        return [""]

    # Extract available options
    course_opts = get_exact_dataset_options('Course')
    edu_opts = get_exact_dataset_options('Education')
    yog_opts = get_exact_dataset_options('Year of Graduation')
    sem_opts = get_exact_dataset_options('Semester')
    stream_opts = get_exact_dataset_options('Stream')
    track_opts = get_exact_dataset_options('Track Interested')
    mode_opts = get_exact_dataset_options('Mode of Program Joined')
    loc_opts = get_exact_dataset_options('Program Location')
    ind_opts = get_exact_dataset_options('Induction session')
    city_opts = get_exact_dataset_options('City')
    state_opts = get_exact_dataset_options('Mailing State')
    country_opts = get_exact_dataset_options('Mailing Country')
    pay_mode_opts = get_exact_dataset_options('Payment_mode')
    source_opts = get_exact_dataset_options('Source of lead')

    # Wizard State Initialization
    if "candidate_step" not in st.session_state:
        st.session_state.candidate_step = 1
    if "candidate_form_data" not in st.session_state:
        st.session_state.candidate_form_data = {}

    def get_index(options_list, val):
        return options_list.index(val) if val in options_list else 0

    st.markdown(f'<div style="text-align: right; color: #8b5cf6; font-weight: bold; margin-bottom: 10px;">Step {st.session_state.candidate_step} of 2</div>', unsafe_allow_html=True)
    st.progress(st.session_state.candidate_step / 2.0)

    if st.session_state.candidate_step == 1:
        st.markdown('<div class="section-header" style="margin-top:10px;"><h2><i class="fa-solid fa-user" style="color:#6366f1; margin-right:8px;"></i> Step 1: Personal & Academic Profile</h2></div>', unsafe_allow_html=True)

        st.markdown('<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 15px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-id-card" style="margin-right:8px;"></i> Personal Identity</div>', unsafe_allow_html=True)
        col_ident1, col_ident2 = st.columns(2)
        with col_ident1:
            candidate_name = st.text_input("Name *", value=st.session_state.candidate_form_data.get('candidate_name', ""), placeholder="John Doe")
            email = st.text_input("Email *", value=st.session_state.candidate_form_data.get('email', ""), placeholder="john@example.com")
            contact_phone = st.text_input("Phone *", value=st.session_state.candidate_form_data.get('contact_phone', ""), placeholder="XXXXX XXXXX")
        with col_ident2:
            contact_id = st.text_input("Contact ID *", value=st.session_state.candidate_form_data.get('contact_id', ""), placeholder="zcrm_XXXX")
            gender_opts = ['Male', 'Female']
            gender = st.selectbox("Gender *", gender_opts, index=get_index(gender_opts, st.session_state.candidate_form_data.get('gender')))
            experience_years = st.number_input("Work Experience (Years) *", min_value=0.0, max_value=50.0, value=st.session_state.candidate_form_data.get('experience_years', 0.0), step=0.5, format="%.1f")

        st.markdown('<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-graduation-cap" style="margin-right:8px;"></i> Enrollment & Academic Details</div>', unsafe_allow_html=True)
        col_edu1, col_edu2 = st.columns(2)
        with col_edu1:
            course = st.selectbox("Course Domain Selection *", course_opts, index=get_index(course_opts, st.session_state.candidate_form_data.get('course')), key="inp_course_live")
            education = st.selectbox("Education Background *", edu_opts, index=get_index(edu_opts, st.session_state.candidate_form_data.get('education')))
        with col_edu2:
            year_of_graduation = st.selectbox("Year of Graduation *", yog_opts, index=get_index(yog_opts, st.session_state.candidate_form_data.get('year_of_graduation')))
            semester = st.selectbox("Current Semester *", sem_opts, index=get_index(sem_opts, st.session_state.candidate_form_data.get('semester')))

        st.markdown('<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-pen-to-square" style="margin-right:8px;"></i> Evaluation Notes</div>', unsafe_allow_html=True)
        background_override = st.text_area("Feedback / Background Override Notes *", value=st.session_state.candidate_form_data.get('background_override', ""), placeholder="Add unique profile feedback notes for categorization...")

        st.markdown("<br>", unsafe_allow_html=True)

        default_total_fee = 0.0
        if 'Course' in df.columns and 'Total_Amount' in df.columns:
            matched_amounts = df[df['Course'] == course]['Total_Amount'].dropna()
            if not matched_amounts.empty:
                try:
                    default_total_fee = float(matched_amounts.mode().iloc[0])
                except Exception:
                    default_total_fee = float(matched_amounts.mean())

        c1, c2 = st.columns([4, 1])
        with c2:
            if st.button("Next", type="primary", icon=":material/arrow_forward:", use_container_width=True):
                # Simple validation of Step 1 fields
                if not candidate_name.strip() or not contact_id.strip() or not email.strip() or not contact_phone.strip() or not background_override.strip():
                    st.error("Submission Denied: All text fields must be fully populated.")
                else:
                    st.session_state.candidate_form_data.update({
                        'candidate_name': candidate_name,
                        'contact_id': contact_id,
                        'email': email,
                        'gender': gender,
                        'contact_phone': contact_phone,
                        'experience_years': experience_years,
                        'course': course,
                        'default_total_fee': default_total_fee,
                        'education': education,
                        'year_of_graduation': year_of_graduation,
                        'semester': semester,
                        'background_override': background_override
                    })
                    st.session_state.candidate_step = 2
                    st.rerun()

    elif st.session_state.candidate_step == 2:
        st.markdown('<div class="section-header" style="margin-top:10px;"><h2><i class="fa-solid fa-graduation-cap" style="color:#34d399; margin-right:8px;"></i> Step 2: Enrollment & Billing Details</h2></div>', unsafe_allow_html=True)

        course = st.session_state.candidate_form_data.get('course', course_opts[0])
        default_total_fee = st.session_state.candidate_form_data.get('default_total_fee', 0.0)

        st.markdown('<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-graduation-cap" style="margin-right:8px;"></i> Course & Track Configuration</div>', unsafe_allow_html=True)
        st.info(f"Selected Course: **{course}**")
        col_course1, col_course2 = st.columns(2)
        with col_course1:
            stream = st.selectbox("Interested Stream *", stream_opts, index=get_index(stream_opts, st.session_state.candidate_form_data.get('stream')))
            track_interested = st.selectbox("Track Customization *", track_opts, index=get_index(track_opts, st.session_state.candidate_form_data.get('track_interested')))
        with col_course2:
            program_mode = st.selectbox("Mode of Program Joined *", mode_opts, index=get_index(mode_opts, st.session_state.candidate_form_data.get('program_mode')))
            program_location = st.selectbox("Program Location *", loc_opts, index=get_index(loc_opts, st.session_state.candidate_form_data.get('program_location')))
            
        col_batch1, col_batch2 = st.columns(2)
        with col_batch1:
            batch_assigned = st.text_input("Batch Assigned *", value=st.session_state.candidate_form_data.get('batch_assigned', ""), placeholder="Aug 2026")

        st.markdown('<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-location-dot" style="margin-right:8px;"></i> Regional Details</div>', unsafe_allow_html=True)
        col_reg1, col_reg2, col_reg3 = st.columns(3)
        with col_reg1:
            city = st.selectbox("City *", city_opts, index=get_index(city_opts, st.session_state.candidate_form_data.get('city')))
        with col_reg2:
            mailing_state = st.selectbox("State *", state_opts, index=get_index(state_opts, st.session_state.candidate_form_data.get('mailing_state')))
        with col_reg3:
            mailing_country = st.selectbox("Country *", country_opts, index=get_index(country_opts, st.session_state.candidate_form_data.get('mailing_country')))

        st.markdown('<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-wallet" style="margin-right:8px;"></i> Billing & Finance Details</div>', unsafe_allow_html=True)
        col_fin1, col_fin2 = st.columns(2)
        with col_fin1:
            payment_date = st.date_input("Payment Date", value=st.session_state.candidate_form_data.get('payment_date'), key="inp_pay_date")
            pay_mode_opts = get_exact_dataset_options('Payment_mode')
            payment_mode = st.selectbox("Payment Mode", pay_mode_opts, index=get_index(pay_mode_opts, st.session_state.candidate_form_data.get('payment_mode')))
            invoice_status = st.selectbox("Invoice Generated?", ["No", "Yes"], index=["No", "Yes"].index(st.session_state.candidate_form_data.get('invoice_status', 'No')) if st.session_state.candidate_form_data.get('invoice_status') in ["No", "Yes"] else 0)
        with col_fin2:
            paid_amount = st.number_input("Paid Amount", min_value=0.0, value=st.session_state.candidate_form_data.get('paid_amount', 0.0), step=100.0, format="%.2f")
            total_amount = st.number_input("Total Amount", min_value=0.0, value=st.session_state.candidate_form_data.get('total_amount', default_total_fee), step=100.0, format="%.2f")

        st.markdown('<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-bullseye" style="margin-right:8px;"></i> Onboarding & Lead Details</div>', unsafe_allow_html=True)
        col_onb1, col_onb2, col_onb3 = st.columns(3)
        with col_onb1:
            source_of_lead = st.selectbox("Source of Lead", source_opts, index=get_index(source_opts, st.session_state.candidate_form_data.get('source_of_lead')))
        with col_onb2:
            feedback_opts = ["Positive", "Negative", "Neutral"]
            feedback_status = st.selectbox("Candidate Intake Feedback *", feedback_opts, index=get_index(feedback_opts, st.session_state.candidate_form_data.get('feedback_status')))
        with col_onb3:
            induction_session = st.selectbox("Induction Session *", ind_opts, index=get_index(ind_opts, st.session_state.candidate_form_data.get('induction_session')))

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        col_chk1, col_chk2 = st.columns(2)
        with col_chk1:
            test_cleared = st.checkbox("Passed Required Test Engine", value=st.session_state.candidate_form_data.get('test_cleared', False))
        with col_chk2:
            followup_sent = st.checkbox("Sent Initial Followup Email", value=st.session_state.candidate_form_data.get('followup_sent', False))

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,3,1])
        with c1:
            if st.button("Back", icon=":material/arrow_back:", use_container_width=True):
                # Save current Step 2 fields to session state
                st.session_state.candidate_form_data.update({
                    'course': course,
                    'stream': stream,
                    'track_interested': track_interested,
                    'program_mode': program_mode,
                    'program_location': program_location,
                    'batch_assigned': batch_assigned,
                    'city': city,
                    'mailing_state': mailing_state,
                    'mailing_country': mailing_country,
                    'payment_date': payment_date,
                    'payment_mode': payment_mode,
                    'paid_amount': paid_amount,
                    'total_amount': total_amount,
                    'source_of_lead': source_of_lead,
                    'feedback_status': feedback_status,
                    'induction_session': induction_session,
                    'invoice_status': invoice_status,
                    'test_cleared': test_cleared,
                    'followup_sent': followup_sent
                })
                st.session_state.candidate_step = 1
                st.rerun()
        with c3:
            submit_btn = st.button("Create Profile", type="primary", icon=":material/check:", use_container_width=True)

        if submit_btn:
            # Stage all data
            fd = st.session_state.candidate_form_data
            candidate_name = fd.get('candidate_name', '')
            contact_id = fd.get('contact_id', '')
            email = fd.get('email', '')
            contact_phone = fd.get('contact_phone', '')
            gender = fd.get('gender', '')
            experience_years = fd.get('experience_years', 0.0)
            education = fd.get('education', '')
            year_of_graduation = fd.get('year_of_graduation', '')
            semester = fd.get('semester', '')
            background_override = fd.get('background_override', '')

            # Simple validation of all fields
            if not candidate_name.strip() or not contact_id.strip() or not email.strip() or not contact_phone.strip() or not batch_assigned.strip() or not background_override.strip():
                st.error("Submission Denied: All required fields must be fully valid and populated.")
                return

                # ── 1. Cleanly Normalize the Streamlit Email ──
            logged_in_email = str(st.session_state.get('user_email', '')).strip().lower()

                # ── 2. Structural Verification Lookup (Python-Driven) ──
            resolved_owner = None

            try:
                mapping_res = supabase.table("salesperson_mappings").select(
                        "salesperson_email, legacy_label").execute()

                if mapping_res.data:
                    for row in mapping_res.data:
                        db_email = str(row.get("salesperson_email", "")).strip().lower()
                        if db_email == logged_in_email:
                            resolved_owner = row.get("legacy_label")
                            break

                if not resolved_owner:
                    st.error(
                            f" Critical Match Error: '{logged_in_email}' was not found in the salesperson mapping records.")
                    return
                else:
                    st.toast(f" Live Database Match Found: {resolved_owner}", icon=":material/thumb_up:")

            except Exception as lookup_err:
                st.error(f"Mapping Database Communication Interruption: {lookup_err}")
                return

            formatted_payment_date = str(payment_date) if payment_date is not None else None

            candidate_payload = {
                "candidate_name": candidate_name.strip(),
                "contact_id": contact_id.strip(),
                "email": email.strip().lower(),
                "contact_phone": contact_phone.strip(),
                "gender": gender,
                "education": education,
                "Year of Graduation": year_of_graduation,
                "Semester": semester,
                "city": city,
                "mailing_state": mailing_state,
                "mailing_country": mailing_country,
                "course": course,
                "stream": stream,
                "track_interested": track_interested,
                "batch_assigned": batch_assigned.strip(),
                "program_mode": program_mode,
                "program_location": program_location,
                "induction_session": induction_session,
                "background_override": background_override.strip(),
                "csv_contact_owner": resolved_owner,
                "Payment_Date": formatted_payment_date,
                "Payment_mode": payment_mode if payment_mode != "" else None,
                "Paid_amount": float(paid_amount),
                "Total_Amount": float(total_amount),
                "Source of Lead": source_of_lead if source_of_lead != "" else None,
                "Feedback": feedback_status,
                "Invoice": invoice_status,
                "Experience": str(experience_years),
                "Test": test_cleared,
                "Followup Email": followup_sent
            }

            try:
                with st.spinner("Synchronizing record with Supabase CRM Database..."):
                    supabase.table("candidates").insert(candidate_payload).execute()
                    st.success(f"Profile for **{candidate_name}** has been successfully generated and linked to {resolved_owner}.")
                    # Clear session state data on success
                    st.session_state.candidate_form_data = {}
                    st.session_state.candidate_step = 1
            except Exception as e:
                st.error(f"Ingestion Interruption: {e}")

# ─────────────────────────────────────────────
# PAGE 6 — CRM NOTES ANALYSIS
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
            merged = df[['Contact Id','Status']].merge(notes, left_on='Contact Id', right_on='Parent ID.id', how='right')
            merged['Status'] = merged['Status'].map({'Joined':'Active', 'Churned':'Churned'}).fillna('Unknown')
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
# PAGE 7 — PAYMENT ANALYSIS
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
    paid_count  = df_inv['Paid_amount'].sum()
    sent_count  = df_inv['Total_Amount'].sum()
    no_inv      = len(df_inv[df_inv['Invoice'].isin(['No', 'No Invoice', 'Nan'])])
    paid_rate   = paid_count / sent_count * 100 if sent_count > 0 else 0

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
            <div class="kpi-title">Total Amount</div>
            <div class="kpi-value" style="color:#fbbf24; font-size:26px;">{sent_count}</div>
            <div class="kpi-sub">Expected payment</div></div>""", unsafe_allow_html=True)
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
    inv_churn = df_inv.groupby(['Invoice', 'Status']).size().reset_index(name='Count')
    inv_churn['Status'] = inv_churn['Status'].map({'Joined':'Active', 'Churned':'Churned'})
    fig3 = px.bar(inv_churn, x='Invoice', y='Count', color='Status',
                  color_discrete_map={'Active':COLOR_ACTIVE,'Churned':COLOR_CHURN},
                  barmode='group', text='Count')
    fig3.update_traces(textfont_size=12, textposition='outside')
    fig3.update_layout(**theme(height=360))
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 8 — LIVE PREDICTOR WITH SUPABASE TELEMETRY
# ─────────────────────────────────────────────
def page_live_predictor(df, model_data, supabase):
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-robot"></i> Live Churn Predictor</h1>
        <p>Dynamic AI predictor backed by machine learning models and real-time database syncing layers.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_data is None:
        st.error("Could not load churn_prediction_model.pkl")
        return

    available_models = model_data.get('available_models', {})
    if not isinstance(available_models, dict) or (hasattr(available_models, 'empty') and available_models.empty):
        available_models = {model_data.get('model_display_name', 'Default Model'): model_data.get('model')}

    # Wizard State Initialization
    if 'predictor_step' not in st.session_state:
        st.session_state.predictor_step = 1
    if 'predictor_data' not in st.session_state:
        st.session_state.predictor_data = {}
    if 'last_searched_email' not in st.session_state:
        st.session_state.last_searched_email = None

    def update_data(**kwargs):
        st.session_state.predictor_data.update(kwargs)

    def next_step():
        st.session_state.predictor_step += 1

    def prev_step():
        st.session_state.predictor_step -= 1

    st.markdown(
        f'<div style="text-align: right; color: #8b5cf6; font-weight: bold; margin-bottom: 10px;">Step {st.session_state.predictor_step} of 3</div>',
        unsafe_allow_html=True)
    st.progress(st.session_state.predictor_step / 3.0)

    llm_options = ["Gemini 2.5 Flash", "Groq (Llama 3)", "Hugging Face (Mistral)"]
    ml_options = list(available_models.keys())
    all_options = llm_options + ml_options

    pd_state = st.session_state.predictor_data

    # Extract model artifacts from model_data
    final_model = model_data.get('model')
    feature_columns = model_data.get('feature_columns', [])
    preprocessor = model_data.get('preprocessor', {})
    categorical_features = model_data.get('categorical_features', [])
    balance_method = model_data.get('balance_method', 'none')

    # Filter for valid Status values (keeping original case checking intact)
    status_mask = df['Status'].astype(str).str.lower().isin(['churned', 'joined'])
    filtered_df = df[status_mask].copy()

    def get_index(options_list, val):
        if val is not None and str(val).strip() in [str(o).strip() for o in options_list]:
            # Match accurately ignoring subtle structural spacing
            for idx, opt in enumerate(options_list):
                if str(opt).strip().lower() == str(val).strip().lower():
                    return idx
        return 0

    if st.session_state.predictor_step == 1:
        st.markdown('<div class="section-header"><h2>Core Details & AI Settings</h2></div>', unsafe_allow_html=True)
        col_ai, col_ident, col_sync = st.columns([1.5, 2, 0.8], vertical_alignment="bottom")

        with col_ai:
            selected_model_name = st.selectbox("Select AI Model", options=all_options, index=get_index(all_options,
                                                pd_state.get('selected_model_name', all_options[0])),
                                                help="Choose which trained algorithm or AI to use for the prediction.")

        with col_ident:
            c_email = st.text_input("Candidate Email Id",
                                    value=pd_state.get('c_email', ""), placeholder="candidate_audit@domain.com",
                                    help="The evaluation history record will log to the production server linked under this index.")

        # ─────────────────────────────────────────────
        # PROFILE LOOKUP & AUTOFILL LOGIC
        # ─────────────────────────────────────────────
        with col_sync:
            sync_triggered = st.button("Sync Profile", type="primary" , use_container_width=True, icon=":material/sync:")

        if sync_triggered and c_email:
            cleaned_target_email = str(c_email).strip().lower()
            # Match directly inside the df table
            matched_candidate = df[df['Email ID'].astype(str).str.strip().str.lower() == cleaned_target_email]

            if not matched_candidate.empty:
                candidate_row = matched_candidate.iloc[0]

                # Maps your visual fields directly back to the database record items safely
                autofill_payload = {
                    'c_email': str(candidate_row.get('email', c_email)),
                    'Gender': candidate_row.get('Gender', None),
                    'Education': candidate_row.get('Education', None),
                    'Course': candidate_row.get('Course', None),
                    'Stream': candidate_row.get('Stream', None),
                    'Track Interested': candidate_row.get('Track Interested', None),
                    'Mode of Program Joined': candidate_row.get('Mode of Program Joined', None),
                    'Batch Assigned': candidate_row.get('Batch Assigned', None),
                    'Source of lead': candidate_row.get('Source of lead', None),
                    'City': candidate_row.get('City', None),
                    'Mailing State': candidate_row.get('Mailing State', None),
                    'Mailing Country': candidate_row.get('Mailing Country', None),
                    'Program Location': candidate_row.get('Program Location', None),
                    'Invoice': candidate_row.get('Invoice', None),
                    'Payment_mode': candidate_row.get('Payment_mode', None),
                    'Total_Amount': int(candidate_row.get('Total_Amount')) if pd.notna(
                        candidate_row.get('Total_Amount')) else 25000,
                    'Paid_amount': int(candidate_row.get('Paid_amount')) if pd.notna(
                        candidate_row.get('Paid_amount')) else 10000,
                    'Induction session': candidate_row.get('Induction session', None),
                    'Feedback': candidate_row.get('Feedback', None),
                    'Test': candidate_row.get('Test', None),
                    'Followup Email': candidate_row.get('Followup Email', None),
                    'Semester': int(candidate_row.get('Semester')) if pd.notna(candidate_row.get('Semester')) else 3,
                    'Year of Graduation': int(candidate_row.get('Year of Graduation')) if pd.notna(
                        candidate_row.get('Year of Graduation')) else 2026,
                    'Experience': int(candidate_row.get('Experience')) if pd.notna(
                        candidate_row.get('Experience')) else 3,
                }

                update_data(**autofill_payload)
                st.toast(f"Profile synchronized successfully for {c_email}!", icon=":material/person_check:")
                st.rerun()
            else:
                st.toast("No existing record found for this email. Continuing with manual entry.",
                         icon=":material/person_search:")

        st.markdown(
            f"**Model Context:** {model_data.get('model_display_name', 'Unknown')}  •  **Balancing Matrix:** {format_balance_method(balance_method)}")

        st.markdown(
            '<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-user" style="margin-right:8px;"></i> Candidate Demographics</div>',
            unsafe_allow_html=True)
        col_dem1, col_dem2 = st.columns(2)
        with col_dem1:
            gender_opts = sorted(df['Gender'].dropna().unique())
            gender = st.selectbox("Gender", gender_opts, index=get_index(gender_opts, pd_state.get('Gender')))
        with col_dem2:
            edu_opts = sorted(df['Education'].dropna().unique())
            education = st.selectbox("Education Background", edu_opts,
                                     index=get_index(edu_opts, pd_state.get('Education')))

        st.markdown(
            '<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-graduation-cap" style="margin-right:8px;"></i> Course & Enrollment Details</div>',
            unsafe_allow_html=True)
        col_prog1, col_prog2, col_prog3 = st.columns(3)
        with col_prog1:
            course_opts = sorted(df['Course'].dropna().unique())
            course = st.selectbox("Course Selected", course_opts, index=get_index(course_opts, pd_state.get('Course')))
            stream_opts = sorted(df['Stream'].dropna().unique())
            stream = st.selectbox("Stream", stream_opts, index=get_index(stream_opts, pd_state.get('Stream')))
        with col_prog2:
            track_opts = sorted(df['Track Interested'].dropna().unique())
            track_interested = st.selectbox("Track Interest", track_opts,
                                            index=get_index(track_opts, pd_state.get('Track Interested')))
            mode_opts = sorted(df['Mode of Program Joined'].dropna().unique())
            mode = st.selectbox("Mode of Program Joined", mode_opts,
                                index=get_index(mode_opts, pd_state.get('Mode of Program Joined')))
        with col_prog3:
            ba_opts = sorted(df['Batch Assigned'].dropna().unique())
            batch_assigned = st.selectbox("Batch Assigned", ba_opts,
                                          index=get_index(ba_opts, pd_state.get('Batch Assigned')))
            source_opts = sorted(df['Source of lead'].dropna().unique())
            source = st.selectbox("Source of Lead", source_opts,
                                  index=get_index(source_opts, pd_state.get('Source of lead')))

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([4, 1])
        with c2:
            if st.button("Next", type="primary", icon=":material/arrow_forward:", use_container_width=True):
                update_data(selected_model_name=selected_model_name, c_email=c_email, Gender=gender, Course=course,
                            **{"Source of lead": source}, Stream=stream, **{"Track Interested": track_interested},
                            **{"Mode of Program Joined": mode}, Education=education,
                            **{"Batch Assigned": batch_assigned})
                next_step()
                st.rerun()

    elif st.session_state.predictor_step == 2:
        st.markdown('<div class="section-header"><h2>Additional & Financial Details</h2></div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 15px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-location-dot" style="margin-right:8px;"></i> Location Details</div>',
            unsafe_allow_html=True)
        col_loc1, col_loc2 = st.columns(2)
        with col_loc1:
            city_opts = sorted(df['City'].dropna().unique())
            city = st.selectbox("City", city_opts, index=get_index(city_opts, pd_state.get('City')))
            state_opts = sorted(df['Mailing State'].dropna().unique())
            mailing_state = st.selectbox("Mailing State", state_opts,
                                         index=get_index(state_opts, pd_state.get('Mailing State')))
        with col_loc2:
            country_opts = sorted(df['Mailing Country'].dropna().unique())
            mailing_country = st.selectbox("Mailing Country", country_opts,
                                           index=get_index(country_opts, pd_state.get('Mailing Country')))
            loc_opts = sorted(df['Program Location'].dropna().unique())
            program_location = st.selectbox("Program Location", loc_opts,
                                            index=get_index(loc_opts, pd_state.get('Program Location')))

        st.markdown(
            '<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-wallet" style="margin-right:8px;"></i> Financial & Billing Details</div>',
            unsafe_allow_html=True)
        col_fin1, col_fin2 = st.columns(2)
        with col_fin1:
            inv_opts = sorted(df['Invoice'].dropna().unique())
            invoice = st.selectbox("Invoice Status", inv_opts, index=get_index(inv_opts, pd_state.get('Invoice')))
            pm_opts = sorted(df['Payment_mode'].dropna().unique())
            payment_mode = st.selectbox("Payment Mode", pm_opts, index=get_index(pm_opts, pd_state.get('Payment_mode')))
        with col_fin2:
            total_amount = st.number_input("Total Fee Amount", min_value=0, value=pd_state.get('Total_Amount', 25000))
            paid_amount = st.number_input("Paid Fee Amount", min_value=0, value=pd_state.get('Paid_amount', 10000))

        st.markdown(
            '<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-bullseye" style="margin-right:8px;"></i> Onboarding & Engagement</div>',
            unsafe_allow_html=True)
        col_eng1, col_eng2 = st.columns(2)
        with col_eng1:
            ind_opts = sorted(df['Induction session'].dropna().unique())
            induction_session = st.selectbox("Induction Session Attendance", ind_opts,
                                             index=get_index(ind_opts, pd_state.get('Induction session')))
            fb_opts = sorted(df['Feedback'].dropna().unique())
            feedback = st.selectbox("Candidate Intake Feedback", fb_opts,
                                    index=get_index(fb_opts, pd_state.get('Feedback')))
        with col_eng2:
            tt_opts = sorted(df['Test'].dropna().unique())
            test_taken = st.selectbox("Initial Test Completed?", tt_opts,
                                      index=get_index(tt_opts, pd_state.get('Test')))
            fe_opts = sorted(df['Followup Email'].dropna().unique())
            followup_email = st.selectbox("Follow-up Email Acknowledged?", fe_opts,
                                          index=get_index(fe_opts, pd_state.get('Followup Email')))

        st.markdown(
            '<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-graduation-cap" style="margin-right:8px;"></i> Academic & Experience Details</div>',
            unsafe_allow_html=True)
        col_edu1, col_edu2 = st.columns(2)
        with col_edu1:
            semester = st.number_input("Current Semester", 0, 10, pd_state.get('Semester', 3))
            year_of_graduation = st.number_input("Year of Graduation (0 if not graduated)", min_value=0, max_value=2050,
                                                 value=pd_state.get('Year of Graduation', 2026))
        with col_edu2:
            experience = st.number_input("Professional Experience (years)", 0, 30, pd_state.get('Experience', 3))

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 3, 1])
        with c1:
            if st.button("Back", icon=":material/arrow_back:", use_container_width=True):
                update_data(City=city, **{"Mailing State": mailing_state}, **{"Mailing Country": mailing_country},
                            Invoice=invoice, **{"Program Location": program_location},
                            **{"Induction session": induction_session}, Feedback=feedback, Payment_mode=payment_mode,
                            Semester=semester, **{"Year of Graduation": year_of_graduation}, Experience=experience,
                            Total_Amount=total_amount, Paid_amount=paid_amount, Test=test_taken,
                            **{"Followup Email": followup_email})
                prev_step()
                st.rerun()
        with c3:
            if st.button("Next", type="primary", icon=":material/arrow_forward:", use_container_width=True):
                update_data(City=city, **{"Mailing State": mailing_state}, **{"Mailing Country": mailing_country},
                            Invoice=invoice, **{"Program Location": program_location},
                            **{"Induction session": induction_session}, Feedback=feedback, Payment_mode=payment_mode,
                            Semester=semester, **{"Year of Graduation": year_of_graduation}, Experience=experience,
                            Total_Amount=total_amount, Paid_amount=paid_amount, Test=test_taken,
                            **{"Followup Email": followup_email})
                next_step()
                st.rerun()

    elif st.session_state.predictor_step == 3:
        st.markdown('<div class="section-header"><h2>Call Inputs & Prediction</h2></div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 15px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-phone" style="margin-right:8px;"></i> Call Conversation Signals</div>',
            unsafe_allow_html=True)
        col_sig1, col_sig2 = st.columns(2)
        with col_sig1:
            not_interested = st.checkbox("No Interest in Course", value=pd_state.get('not_interested', False))
            unreachable_not_connected = st.checkbox("No Response / Unreachable",
                                                    value=pd_state.get('unreachable_not_connected', False))
            joined_competitor = st.checkbox("Joined in another institution",
                                            value=pd_state.get('joined_competitor', False))
            decision_pending = st.checkbox("Technical Discussion", value=pd_state.get('decision_pending', False))
        with col_sig2:
            financial_issue = st.checkbox("Course fees not affordable", value=pd_state.get('financial_issue', False))
            already_working = st.checkbox("Got placed", value=pd_state.get('already_working', False))
            looking_for_job = st.checkbox("Job hunting, not internship", value=pd_state.get('looking_for_job', False))
            join_later = st.checkbox("Wants to join later / Postpone", value=pd_state.get('join_later', False))  # Added

        st.markdown(
            '<div style="font-size:15px; font-weight:700; color:#38bdf8; margin: 25px 0 10px 0; text-transform:uppercase; letter-spacing:0.5px;"><i class="fa-solid fa-pen-to-square" style="margin-right:8px;"></i> Call Notes & Remarks</div>',
            unsafe_allow_html=True)
        call_transcript = st.text_area("Call Transcript (optional)", value=pd_state.get('call_transcript', ""),
                                       max_chars=2000, placeholder="Paste full call transcript...")
        call_remarks = ""
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            if st.button("Back", icon=":material/arrow_back:", use_container_width=True):
                update_data(not_interested=not_interested, unreachable_not_connected=unreachable_not_connected,
                            joined_competitor=joined_competitor, financial_issue=financial_issue,
                            already_working=already_working, looking_for_job=looking_for_job,
                            decision_pending=decision_pending, call_remarks=call_remarks,
                            call_transcript=call_transcript)
                prev_step()
                st.rerun()
        with c3:
            predict_btn = st.button("Predict Churn Risk", icon=":material/online_prediction:", use_container_width=True,
                                    type="primary")

        if predict_btn:
            pd_state.update(not_interested=not_interested, unreachable_not_connected=unreachable_not_connected,
                            joined_competitor=joined_competitor, financial_issue=financial_issue,
                            already_working=already_working, looking_for_job=looking_for_job,
                            decision_pending=decision_pending, join_later=join_later, call_remarks=call_remarks,
                            call_transcript=call_transcript)

            total_amount = pd_state.get('Total_Amount', 25000)
            paid_amount = pd_state.get('Paid_amount', 10000)
            derived_paid_rate = (paid_amount / total_amount) if total_amount > 0 else 0.0
            selected_model_name = pd_state.get('selected_model_name', all_options[0])
            is_llm = selected_model_name in llm_options
            c_email = pd_state.get('c_email', 'candidate_audit@domain.com')
            semester = pd_state.get('Semester', 3)
            year_of_graduation = pd_state.get('Year of Graduation', 2026)
            experience = pd_state.get('Experience', 3)
            course = pd_state.get('Course', 'Unknown')
            track_interested = pd_state.get('Track Interested', 'Unknown')
            city = pd_state.get('City', 'Unknown')
            mailing_state = pd_state.get('Mailing State', 'Unknown')
            mailing_country = pd_state.get('Mailing Country', 'Unknown')
            source = pd_state.get('Source of lead', 'Unknown')
            gender = pd_state.get('Gender', 'Unknown')
            test_taken = pd_state.get('Test', 'Unknown')
            followup_email = pd_state.get('Followup Email', 'Unknown')
            invoice = pd_state.get('Invoice', 'Unknown')
            mode = pd_state.get('Mode of Program Joined', 'Unknown')
            program_location = pd_state.get('Program Location', 'Unknown')
            education = pd_state.get('Education', 'Unknown')
            batch_assigned = pd_state.get('Batch Assigned', 'Unknown')
            stream = pd_state.get('Stream', 'Unknown')
            induction_session = pd_state.get('Induction session', 'Unknown')
            feedback = pd_state.get('Feedback', 'Unknown')
            payment_mode = pd_state.get('Payment_mode', 'Unknown')

            input_data = {
                'Semester': semester, 'Year of Graduation': year_of_graduation, 'Experience': experience,
                'Total_Amount': total_amount, 'Paid_amount': paid_amount, 'Paid_Rate': derived_paid_rate,
                'Course': course, 'Track Interested': track_interested, 'City': city, 'Mailing State': mailing_state,
                'Mailing Country': mailing_country, 'Source of lead': source, 'Gender': gender, 'Test': test_taken,
                'Followup Email': followup_email, 'Invoice': invoice, 'Mode of Program Joined': mode,
                'Program Location': program_location, 'Education': education, 'Batch Assigned': batch_assigned,
                'Stream': stream, 'Induction session': induction_session, 'Feedback': feedback,
                'Payment_mode': payment_mode, 'email': c_email
            }

            if not is_llm:
                model_df_input = input_data.copy()
                model_df_input['role'] = "professional" if experience > 0 else "student"
                model_df_input['background'] = "tech" if "tech" in str(education).lower() else "non tech"

                # Map the checkboxes directly to the numerical 1/0 vectors expected by X
                model_df_input['not_interested'] = int(not_interested)
                model_df_input['joined_competitor'] = int(joined_competitor)
                model_df_input['decision_pending'] = int(decision_pending)
                model_df_input['already_working'] = int(already_working)
                model_df_input['looking_for_job'] = int(looking_for_job)
                model_df_input['financial_issue'] = int(financial_issue)
                model_df_input['join_later'] = int(join_later)

                input_df = pd.DataFrame([model_df_input])

                for col in categorical_features:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(str)

                processed_input = preprocessor.transform(input_df)
                if hasattr(processed_input, 'toarray'):
                    processed_input_df = pd.DataFrame(processed_input.toarray(), columns=feature_columns)
                else:
                    processed_input_df = pd.DataFrame(processed_input, columns=feature_columns)
                processed_input_df = processed_input_df.reindex(columns=feature_columns, fill_value=0)

                final_model = available_models.get(selected_model_name,
                                                   available_models.get(list(available_models.keys())[0]))

            try:
                if is_llm:
                    from llm_integration import get_llm_prediction
                    with st.spinner(f"Querying {selected_model_name} AI Strategist..."):
                        llm_result = get_llm_prediction(selected_model_name, input_data)
                    prob = float(llm_result.get("churn_probability", 50.0)) / 100.0
                    pred = 1 if prob >= 0.5 else 0
                    llm_reason = llm_result.get("reason", "Inference processed successfully via LLM rules engine.")
                    llm_retention = llm_result.get("retention_strategy",
                                                   "Maintain standard operations tracking playbook.")
                    model_display = selected_model_name
                else:
                    pred_raw = final_model.predict(processed_input_df)
                    prob_raw = final_model.predict_proba(processed_input_df)[:, 1]
                    pred = int(np.asarray(pred_raw).flatten()[0])
                    prob = float(np.asarray(prob_raw).flatten()[0])
                    prob = max(0.0, min(1.0, prob))
                    llm_reason = "Derived dynamically using feature importance patterns across matrix weights."
                    llm_retention = "Initiate standardized recovery sequences based on risk vectors."
                    model_display = model_data.get('model_display_name', selected_model_name)

                # --- SUPABASE DATA WRITER ---
                user_ip = st.context.ip_address if st.context.ip_address else "127.0.0.1"

                from streamlit.runtime.scriptrunner import get_script_run_ctx
                ctx = get_script_run_ctx()
                current_session_id = ctx.session_id if (ctx and hasattr(ctx, 'session_id')) else "unknown"

                resolved_candidate_id = None
                if supabase is not None and c_email:
                    try:
                        candidate_lookup = supabase.table("candidates").select("id").eq("email", c_email).limit(
                            1).execute()
                        if candidate_lookup.data:
                            resolved_candidate_id = candidate_lookup.data[0]["id"]
                    except Exception:
                        pass

                prev_pred_id = None
                if supabase is not None and c_email:
                    try:
                        prev_q = supabase.table("predictions").select("id").eq("email", c_email).order("predicted_at",
                                                                                                       desc=True).limit(
                            1).execute()
                        if prev_q.data:
                            prev_pred_id = prev_q.data[0]["id"]
                    except Exception:
                        pass

                db_record = {
                    "candidate_id": resolved_candidate_id,
                    "semester": semester, "year_of_graduation": year_of_graduation, "experience": experience,
                    "total_amount": float(total_amount), "paid_amount": float(paid_amount), "course": course,
                    "track_interested": track_interested, "city": city, "mailing_state": mailing_state,
                    "mailing_country": mailing_country, "source_of_lead": source, "gender": gender, "invoice": invoice,
                    "mode_of_program": mode, "program_location": program_location, "education": education,
                    "batch_assigned": batch_assigned, "stream": stream, "induction_session": induction_session,
                    "feedback": feedback, "payment_mode": payment_mode, "email": c_email,
                    "predicted_status": "Churned" if pred == 1 else "Joined", "churn_probability": round(prob, 4),
                    "risk_factors": identify_risk_factors(input_data), "model_algorithm": model_display,
                    "predicted_by": st.session_state.get("user_email", "dashboard_agent"),
                    "risk_level": "high" if prob > 0.65 else ("medium" if prob > 0.35 else "low"),
                    "client_ip": user_ip, "previous_prediction_id": prev_pred_id, "session_id": current_session_id,
                    "confidence_score": round(abs(prob - 0.5) * 2, 4), "model_version": "v1.0.0-prod"
                }

                if supabase is not None:
                    with st.spinner("Streaming analytical updates back to ledger database server..."):
                        db_resp = supabase.table("predictions").insert(db_record).select("role", "background",
                                                                                         "paid_rate").execute()
                        db_metrics = db_resp.data[0] if db_resp.data else {}
                else:
                    db_metrics = {}

                # UI Rendering Results Layout
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
                        mode="gauge+number", value=round(prob * 100, 1),
                        title={"text": "Churn Probability", "font": {"color": "#94a3b8", "size": 14}},
                        number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 36}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#475569"},
                            "bar": {"color": "#f87171" if prob > 0.5 else "#34d399", "thickness": 0.3},
                            "bgcolor": "rgba(0,0,0,0)",
                            "steps": [
                                {"range": [0, 30], "color": "rgba(52,211,153,0.15)"},
                                {"range": [30, 60], "color": "rgba(251,191,36,0.15)"},
                                {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
                            ],
                        }
                    ))
                    gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#94a3b8'),
                                        margin=dict(l=20, r=20, t=40, b=20), height=250)
                    st.plotly_chart(gauge, use_container_width=True)

                with r3:
                    ret_role = db_metrics.get('role', 'professional' if experience > 0 else 'student')
                    ret_bg = db_metrics.get('background', 'tech' if "tech" in str(education).lower() else 'non tech')
                    ret_prate = float(db_metrics.get('paid_rate') or derived_paid_rate) * 100

                    action_req = '<i class="fa-solid fa-triangle-exclamation" style="color:#fbbf24"></i> <b style="color:#fbbf24;">Action Required:</b> Immediate intervention.' if pred == 1 else '<i class="fa-solid fa-circle-check" style="color:#34d399"></i> <b style="color:#34d399;">On Track:</b> Monitor.'

                    st.markdown(f"""<div class="candidate-card" style="margin-top:0; height:100%; display:flex; flex-direction:column;">
                                <div style="font-size:13px; font-weight:700; color:#38bdf8; margin-bottom:5px; text-transform:uppercase;">Database Generated Outputs</div>
                                <div style="font-size:12px; color:#cbd5e1; margin-bottom:10px; line-height:1.4;">
                                • Computed Role Category: <b style="color:#a78bfa;">{ret_role}</b><br>
                                • Classified Background Profile: <b style="color:#a78bfa;">{ret_bg}</b><br>
                                • Actual Remittance Value Rate: <b style="color:#a78bfa;">{ret_prate:.1f}%</b>
                                </div>
                                <hr style="border-color:rgba(255,255,255,0.1); margin:4px 0;">
                                <div style="font-size:12px; color:#e2e8f0; margin-bottom:8px;">
                                <span style="color:#64748b; font-weight:600;">Reasoning:</span><br>
                                <i>"{llm_reason}"</i>
                                </div>
                                <div style="font-size:12px; color:#38bdf8; margin-bottom:10px; flex-grow:1;">
                                <span style="color:#64748b; font-weight:600;">Retention Strategy:</span><br>
                                <b>{llm_retention}</b>
                                </div>
                                <hr style="border-color:rgba(255,255,255,0.1); margin:4px 0;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div style="font-size:11px; color:#64748b;">Using {model_display}</div>
                                <div style="font-size:12px;">{action_req}</div>
                                </div>
                                </div>""", unsafe_allow_html=True)

                    st.toast("Telemetry matrix updates synced to Supabase.", icon=":material/database:")
            except Exception as e:
                st.error(f"Prediction Pipeline Faulted: {e}")

# ─────────────────────────────────────────────
# PAGE 9 — MODEL PERFORMANCE
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

    churned_df = df[df['Status'] == 'Churned'].copy()
    active_df = df[df['Status'] == 'Joined'].copy()

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown("""<div class="candidate-card">
                           <div style="font-size:13px; font-weight:700; color:#f87171; margin-bottom:12px;"><i class="fa-solid fa-circle-xmark"></i> Not interested</div>""",
                    unsafe_allow_html=True)
        # Filter for 'not interested' only
        no_interest_mask = churned_df['final_inferred_reason'].str.lower().str.contains('not interested', na=False)
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
        no_resp_mask = churned_df['final_inferred_reason'].str.lower().str.contains('other', na=False)
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
        st.markdown("### Top 15 Feature Importance ")
        fig = px.bar(feat_df, x='Coefficient', y='Feature', orientation='h',
                     color='Coefficient', color_continuous_scale=['#4c1d95','#6366f1','#06b6d4'])
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.update_layout(**theme(height=400, showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance report not found.")

# profile
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
        <p style="margin:5px 0;"><strong>Assigned Access Tier Role:</strong> {st.session_state.get('user_role', 'Salesperson')}</p>
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


# ==============================================================================
# MAIN PAGE ROUTER ENGINE
# ==============================================================================
def main():
    if not st.session_state.get("logged_in", False):
        page_auth()
        return

    # Injected Top Header profile-button layout styles
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

    # 1. Execute early sidebar processing data pass
    page = sidebar()

    # 2. Extract the authenticated User's UUID dynamically from Supabase
    try:
        user_session_info = supabase.auth.get_user(st.session_state.get("access_token"))
        logged_in_user_uuid = user_session_info.user.id

        # Double check sync structure of the session role assignment state object
        if "user_role" not in st.session_state:
            st.session_state.user_role = user_session_info.user.user_metadata.get("role", "Salesperson")
    except Exception:
        logged_in_user_uuid = "00000000-0000-0000-0000-000000000000"

    # 3. Render right-hand side Profile UI switch toggler
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

    # 4. Sidebar state check exits Profile View if a menu link is explicitly clicked
    if st.session_state.get("show_profile", False):
        if st.session_state.get("current_page", "Overview") != st.session_state.get("last_seen_page", page):
            st.session_state.show_profile = False
        else:
            st.session_state.last_seen_page = page
            page_profile()
            return

    st.session_state.last_seen_page = page

    # 5. Storage Loader Process Memory Block
    with st.spinner("Loading data..."):
        try:
            raw_df, raw_notes, source_type = load_data()

            if source_type == "database":
                df, notes = preprocess(raw_df, raw_notes)
            else:
                df, notes = raw_df, raw_notes

            st.session_state['df'] = df
            st.session_state['notes'] = notes

        except Exception as e:
            st.error(f"Could not load data files: {e}")
            st.stop()

    model_path = os.path.join(OUTPUT_DIR, "prediction_model.pkl")
    model_modified_time = os.path.getmtime(model_path) if os.path.exists(model_path) else None
    model_data = load_model()

    # 6. Master Multi-Page Execution Routing Branch with Route-Level Guarding
    user_role = st.session_state.get("user_role", "Salesperson")

    if page == "Overview" and user_role == "Admin":
        page_overview(df, notes)
    elif page == "Candidate Explorer" and user_role == "Admin":
        page_candidate_explorer(df, notes)
    elif page == "Salesperson Analytics" and user_role == "Admin":  # 🌟 ADDED THIS ROUTE
        page_salesperson_stats(df, supabase)
    elif page == "Smart Agent Workspace":
        render_agent_workspace_and_logger(supabase, logged_in_user_uuid)
    elif page == "Add New Candidate":
        render_candidate_entry_form(df, notes)
    elif page == "CRM Notes Analysis" and user_role == "Admin":
        page_notes_analysis(df, notes)
    elif page == "Invoice Analysis" and user_role == "Admin":
        page_payment_analysis(df)
    elif "Predictor" in page:
        page_live_predictor(df, model_data, supabase)
    elif "Model Performance" in page and user_role == "Admin":
        page_model_performance(df, model_data)
    else:
        # Fallback security redirection route catch-all step
        st.warning("You don't have authorization permissions to access this page section.")
        st.session_state.current_page = "Smart Agent Workspace"
        st.rerun()


if __name__ == "__main__":
    main()

