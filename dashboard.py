import numpy as np
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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





 

ARTIFACTS_PATH = './output'

 
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
            ("Notes Analysis", ":material/notes:"),
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


def page_overview():
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-chart-pie"></i> Executive Overview</h1>
        <p>Real-time snapshot of candidate churn status, sources, and engagement metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
     try:
         enrolled = pd.read_csv(os.path.join(ARTIFACTS_PATH, 'enrolled_processed.csv'))
         notes = pd.read_csv(os.path.join(ARTIFACTS_PATH, 'notes_processed.csv'))
         return enrolled, notes
     except FileNotFoundError:
         st.error("Data files not found. Please ensure 'enrolled_processed.csv' and 'notes_processed.csv' are in the 'output' directory.")
         return pd.DataFrame(), pd.DataFrame()
 
    enrolled_df, notes_df = load_data()

# Define conditions for churn
    conditions = [
    (enrolled_df['Program Joined'] != 'Not joined') & (enrolled_df['Invoice'] == 'Yes'), # Non-churned (Active with Invoice)
    (enrolled_df['Program Joined'] == 'Not joined') & (enrolled_df['Invoice'] == 'Yes'), # Churned with Invoice
    (enrolled_df['Invoice'] == 'No') # Candidates without an invoice, or where invoice status is not 'Yes'
]

# Define choices for churn (0 for non-churn, 1 for churn, -1 for other cases)
    choices = [0, 1, -1]

    enrolled_df['churn'] = np.select(conditions, choices, default=-1) # Use -1 to denote cases not explicitly 0 or 1


    if not enrolled_df.empty:
     st.subheader("Key Performance Indicators (KPIs)")
 
     total_candidates_contacted = 46469 # From CRM dataset analysis in notebook
     registered_candidates = enrolled_df[enrolled_df['Invoice'] == 'Yes'].shape[0]
     enrolled_candidates = enrolled_df.shape[0]
     # Calculate based on Program Joined
     joined_candidates_count = enrolled_df[enrolled_df['Program Joined'] != 'Not joined'].shape[0]
     not_joined_candidates_count = enrolled_df[enrolled_df['Program Joined'] == 'Not joined'].shape[0]
 
     # Recalculate churned/active based on the notebook's final logic for churn definition
     # Assuming 'churn' column is in enrolled_df (0=non-churn, 1=churn, -1=not classified)
     churned_candidates = enrolled_df[enrolled_df['churn'] == 1].shape[0]
     active_candidates = enrolled_df[enrolled_df['churn'] == 0].shape[0]
 
     col1, col2, col3, col4, col5 = st.columns(5)
     with col1:
         st.metric(label="Total Contacts (CRM)", value=f"{total_candidates_contacted:,}")
     with col2:
         st.metric(label="Enrolled Records", value=f"{enrolled_candidates:,}")
     with col3:
         st.metric(label="Registered Candidates (Invoice Yes)", value=f"{registered_candidates:,}")
     with col4:
         st.metric(label="Active Candidates (Model Target)", value=f"{active_candidates:,}")
     with col5:
         st.metric(label="Churned Candidates (Model Target)", value=f"{churned_candidates:,}")
 
     st.subheader("Visualizations")
 
#     # Display existing plots from the output directory
     st.markdown("#### Course Distribution")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'course_distribution.png'))
     except FileNotFoundError:
         st.warning("Course distribution plot not found. Please run the notebook to generate it.")
 
     st.markdown("#### Churn vs. Active Candidates Distribution")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'churn_active_candidates_distribution.png'))
     except FileNotFoundError:
         st.warning("Churn vs. Active candidates distribution plot not found. Please run the notebook to generate it.")
 
     st.markdown("#### Candidate Role Distribution")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'role_distribution.png'))
     except FileNotFoundError:
         st.warning("Role distribution plot not found. Please run the notebook to generate it.")
 
     st.markdown("#### Candidate Background Distribution")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'background_distribution.png'))
     except FileNotFoundError:
         st.warning("Background distribution plot not found. Please run the notebook to generate it.")
 
     st.markdown("#### Churn by Source of Lead")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'churn_by_source_of_lead.png'))
     except FileNotFoundError:
         st.warning("Churn by source of lead plot not found. Please run the notebook to generate it.")
 
     st.markdown("#### Churn by Training Mode")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'churn_by_training_mode.png'))
     except FileNotFoundError:
         st.warning("Churn by training mode plot not found. Please run the notebook to generate it.")
 
     st.markdown("#### Program Growth by Invoice Status Across Years")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'program_growth_by_invoice.png'))
     except FileNotFoundError:
         st.warning("Program growth by invoice status plot not found. Please run the notebook to generate it.")
 
    else:
     st.info("No data available to display. Please ensure data processing is complete in the notebook.")


#  pages/2_📝_Notes_Analysis.py
 

def page_note_analysis():
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-clipboard-list"></i> Notes Analysis</h1>
        <p>Explore candidate notes to understand common themes, inferred statuses, and reasons for churn.</p>
    </div>
    """, unsafe_allow_html=True) 
    @st.cache_data
    def load_notes_data():
        try:
             # Ensure the 'notes_processed.csv' contains 'cleaned_content', 'inferred_status', 'inferred_reason'
            notes_df = pd.read_csv(os.path.join(ARTIFACTS_PATH, 'notes_processed.csv'))
         # Convert 'cleaned_content' back to list of words if it was saved as string representation of list
            if 'cleaned_content' in notes_df.columns:
                 notes_df['cleaned_content'] = notes_df['cleaned_content'].apply(eval) # Use eval to convert string list to actual list
            return notes_df
        except FileNotFoundError:
            st.error("Processed notes data not found. Please ensure 'notes_processed.csv' is in the 'output' directory.")
            return pd.DataFrame()
 
    notes_df = load_notes_data()
 
    if not notes_df.empty:
     st.subheader("Top Words in Notes")
     st.write("Understanding the most frequently used words in candidate notes can highlight common themes.")
 
     # Display existing plot
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'top_words_notes.png'))
     except FileNotFoundError:
         st.warning("Top words plot not found. Please run the notebook to generate it.")
 
     st.subheader("Inferred Status Distribution")
     st.write("This section shows the distribution of statuses inferred from the notes (Joined, Not Joined, Unclear, Join Later).")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'final_inferred_status_distribution.png'))
     except FileNotFoundError:
         st.warning("Final inferred status distribution plot not found. Please run the notebook to generate it.")
 
 
     st.subheader("Inferred Reasons for Not Joining")
     st.write("A breakdown of the reasons why candidates did not join, as inferred from their notes.")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'inferred_reasons_notes.png'))
     except FileNotFoundError:
         st.warning("Inferred reasons for not joining plot not found. Please run the notebook to generate it.")
 
     st.subheader("Top Suggested Churn Reasons (Aggregated)")
     st.write("This chart shows the top reasons why candidates might churn based on aggregated notes.")
     try:
         st.image(os.path.join(ARTIFACTS_PATH, 'final_inferred_reason_distribution.png'))
     except FileNotFoundError:
         st.warning("Final inferred reason distribution plot not found. Please run the notebook to generate it.")
 
 
     st.subheader("Explore Keywords in Context")
     st.write("Enter a keyword to see sentences from notes where it appears, providing deeper context.")
 
     search_keyword = st.text_input("Enter keyword (e.g., 'internship', 'job', 'finance')", "internship")
     num_examples = st.slider("Number of example sentences to show", 1, 20, 5)
 
     if search_keyword:
         found_sentences = []
         for sentences_list in notes_df['sentences'].dropna():
             # Ensure sentences_list is actually a list (it might be a string representation if not eval'd)
             if isinstance(sentences_list, str):
                 try:
                     sentences_list = eval(sentences_list)
                 except:
                     continue # Skip if can't evaluate
 
             if isinstance(sentences_list, list):
                 for sentence in sentences_list:
                     if search_keyword.lower() in str(sentence).lower():
                         found_sentences.append(sentence)
 
         if found_sentences:
             st.success(f"Found {len(found_sentences)} sentences containing '{search_keyword}'. Showing top {min(len(found_sentences), num_examples)}:")
             for i, sentence in enumerate(found_sentences[:num_examples]):
                 st.markdown(f"- {sentence}")
         else:
             st.info(f"No sentences found containing '{search_keyword}'.")
    else:
     st.info("No notes data available to display. Please ensure data processing is complete in the notebook.")


# %%writefile pages/3_🔮_Candidate_Churn_Prediction.py

def page_live_predictor():
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-robot"></i> Live Churn Predictor</h1>
        <p>Fill in candidate details to get an instant AI-powered churn risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)
 
    @st.cache_resource
    def load_model_artifacts():
        try:
            with open(os.path.join(ARTIFACTS_PATH, 'churn_prediction_model.pkl'), 'rb') as f:
               model_data = pickle.load(f)
            return model_data
        except FileNotFoundError:
            st.error("Model artifacts not found. Please run the notebook to train and save the model.")
            return None
 
    model_artifacts = load_model_artifacts()
 
    if model_artifacts:
        model = model_artifacts['model']
        scaler = model_artifacts['scaler']
        feature_columns = model_artifacts['feature_columns']
        categorical_features = model_artifacts['categorical_features']
        numerical_features = model_artifacts['numerical_features']
 
        st.subheader("Live Churn Prediction")
        st.write("Enter candidate details below to predict their churn status and get recommendations.")
 
     # Input form for candidate features
        with st.form("churn_prediction_form"):
            st.markdown("##### Personal & Program Details")
            col1, col2, col3 = st.columns(3)
            with col1:
               gender = st.selectbox("Gender", ['Male', 'Female', 'Others'])
               experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=0)
               semester = st.number_input("Semester (if student)", min_value=0, max_value=12, value=0)
            with col2:
               year_of_graduation = st.number_input("Year of Graduation (0 if not graduated)", min_value=0, max_value=2050, value=2024)
               course = st.text_input("Course (e.g., DA, DS, BTECH)", "UNSPECIFIED")
               source_of_lead = st.selectbox("Source of Lead", ['Indeed', 'Reference', 'Digital Marketing (Goat add)', 'Infopark website', 'Direct', 'Bulk Lead (3000)', 'Free internship campaign', 'Digital Marketing (Add On)', 'Others', 'Organic', 'Naukri Paid', 'Website Enquiry', 'Social Media', 'OLX', 'Instagram', 'Justdial', 'Seminar', 'Job Fair', 'Walk-in', 'LinkedIn', 'YouTube', 'Telecalling', 'Google', 'WhatsApp', 'DM', 'Carryover', 'college referal', 'Webinar', 'dm', 'free internship', 'infopark', 'I'])
            with col3:
               track_interested = st.selectbox("Track Interested", ['Data Analytics', 'Python Full stack', 'Datascience/DataAnalytics', 'Datascience', 'MERN Stack', 'Artificial Intelligence', 'Fullstack', 'Data Science & AI', 'DevOps', 'Cloud Computing', 'Digital Marketing', 'Cyber Security', 'Block Chain', 'Machine Learning', 'Big Data', 'Web Development', 'UI/UX Design', 'Flutter', 'Data Science', 'Data Engineering', 'Not mentioned'])
               mode_of_program_joined = st.selectbox("Mode of Program Joined", ['Not mentioned', 'Online', 'Onsite', 'Hybrid', 'online hybrid'])
               batch_assigned_to = st.selectbox("Batch Assigned to", ['Not assigned', '2024-01-01 00:00:00', '2025-01-01 00:00:00', '2026-02-25 00:00:00', 'First Choice', '2026-06-24 00:00:00', '2025-03-01 00:00:00', '2026-12-25 00:00:00'])
 
            st.markdown("##### Other Flags")
            col4, col5 = st.columns(2)
            with col4:
               test_taken = st.checkbox("Test Taken")
               followup_email_sent = st.checkbox("Followup Email Sent")
               invoice_generated = st.checkbox("Invoice Generated")
            with col5:
             # Inferred Reasons (simulated based on typical keywords, for live prediction user might not know this directly)
             # For a real app, these might be derived from other text inputs or internal logic
               st.markdown("*(For simplified demo, select inferred reasons directly)*")
               joined_competitor = st.checkbox("Joined a Competitor (Inferred)")
               already_working = st.checkbox("Already Working/Internship (Inferred)")
               looking_for_job_internship = st.checkbox("Looking for Job/Internship (Inferred)")
               not_interested = st.checkbox("Not Interested (Inferred)")
               financial_issue = st.checkbox("Financial Issue (Inferred)")
               unreachable_not_connected = st.checkbox("Unreachable/Not Connected (Inferred)")
               decision_pending = st.checkbox("Decision Pending (Inferred)")
               no_notes_provided = st.checkbox("No Notes Provided (Inferred)")
 
            submitted = st.form_submit_button("Predict Churn")
 
        if submitted:
         # Create a DataFrame from user inputs
            input_data = pd.DataFrame([{
               'Experience': experience,
               'Semester': semester,
               'Year of Graduation': year_of_graduation,
               'Test': int(test_taken),
               'Followup Email': int(followup_email_sent),
               'Invoice_binary': int(invoice_generated),
               'joined_competitor': int(joined_competitor),
               'already_working': int(already_working),
               'looking_for_job_internship': int(looking_for_job_internship),
               'not_interested': int(not_interested),
               'financial_issue': int(financial_issue),
               'unreachable_not_connected': int(unreachable_not_connected),
               'decision_pending': int(decision_pending),
               'no_notes_provided': int(no_notes_provided),
 
             # Categorical features
               'Source of lead': source_of_lead,
               'Course': course,
               'background': 'UNKNOWN', # Placeholder, ideally derived dynamically from course
               'role': 'Unknown', # Placeholder, ideally derived dynamically from experience/semester/grad_year
               'final_inferred_status': 'Unclear', # Placeholder, as this comes from notes analysis
               'Track Interested': track_interested,
               'Mode of Program Joined': mode_of_program_joined,
               'Batch Assigned to': batch_assigned_to,
               'Gender': gender
            }])
 
         # --- Preprocessing steps for input_data, mirroring the notebook ---
         # 1. Standardize course (simplified for Streamlit demo)
         # This would require replicating the `standardize_course_name` logic here or pre-calculating options.
         # For demo, just use the input 'course' as is, but a full app would need the function.
 
         # 2. Derive 'role' and 'background' (simplified/placeholder)
         # Replicate `assign_role` and `assign_background` from notebook here or use simple mapping.
            def assign_role_streamlit(row):
                if row['Experience'] > 0:
                     return 'Professional'
                elif row['Semester'] > 0:
                     return 'Student'
                elif row['Year of Graduation'] > 0 and row['Experience'] == 0: # Assuming graduated + no experience = idle
                     return 'Idle or Career Gap'
                return 'Unknown'
 
            input_data['role'] = input_data.apply(assign_role_streamlit, axis=1)
 
            def assign_background_streamlit(course_str):
                course_str_upper = str(course_str).upper()
                tech_keywords = ['BTECH', 'BE', 'MTECH', 'BCA', 'MCA', 'BSC-CS', 'MSC-CS', 'IT', 'DA', 'DS', 'AI', 'COMPUTER']
                if any(k in course_str_upper for k in tech_keywords): return 'Tech'
                non_tech_keywords = ['BCOM', 'MCOM', 'BA', 'MA', 'ARTS', 'COMMERCE']
                if any(k in course_str_upper for k in non_tech_keywords): return 'Non-Tech'
                return 'UNKNOWN'
 
            input_data['background'] = input_data['Course'].apply(assign_background_streamlit)
 
         # Create dummy variables for categorical features
         # Need to ensure all possible columns from training are present
         # Create a dummy dataframe with all known feature columns from training data to align columns
            dummy_df = pd.DataFrame(columns=feature_columns)
            processed_input = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)
 
         # Align columns - add missing columns with 0 and remove extra ones
            for col in feature_columns:
                if col not in processed_input.columns:
                    processed_input[col] = 0
            processed_input = processed_input[feature_columns] # Ensure order and presence
 
         # Scale numerical features
            processed_input[numerical_features] = scaler.transform(processed_input[numerical_features])
 
         # --- Prediction ---
            churn_probability = model.predict_proba(processed_input)[:, 1][0]
            churn_prediction = model.predict(processed_input)[0]
 
            st.subheader("Prediction Result")
            if churn_prediction == 1:
                st.error(f"This candidate is predicted to **CHURN** with a probability of {churn_probability:.2f}.")
            else:
                st.success(f"This candidate is predicted to **NOT CHURN** with a probability of {1 - churn_probability:.2f}.")
 
            st.subheader("Suggested Churn Reason & Recommended Action")
         # Replicate the heuristic_recommendation logic from the notebook
            def heuristic_recommendation(reason_label):
                mapping = {
                 'Financial issues': 'Offer flexible payment plans, scholarships, or budget-friendly EMI options and follow up on affordability concerns.',
                 'Not Interested': 'Re-engage with personalized course benefits, clarify learning outcomes, and offer a second consultation call.',
                 'Joined another institution': 'Reach out with retention incentives, compare program strengths, and propose a unique value-added offer.',
                 'Unreachable/Not Connected': 'Increase outreach frequency, confirm contact details, and assign a dedicated counselor for follow-up.',
                 'Already Working/Internship': 'Highlight career advancement opportunities and specialized training. Emphasize how the program complements existing experience.',
                 'Looking for Job/Internship': 'Provide insights into job market trends, showcase success stories, and offer career counseling specific to job/internship search.',
                 'Decision Pending/Discussing': 'Follow up with clarity on program benefits, address family concerns, and offer a consultation with an expert.',
                 'Location Issue': 'Suggest online/remote learning options or alternative program locations if available.',
                 'Time/Schedule Conflict': 'Offer flexible scheduling, part-time options, or self-paced learning modules.',
                 'Interested/Pending': 'Provide more detailed information, address specific queries, and maintain regular follow-up to convert interest.',
                 'Details Shared/Collected': 'Ensure timely processing of candidate details and initiate contact for next steps.',
                 'No Notes Provided': 'Investigate the candidate details further and provide a customized recovery plan based on the latest call context.',
                 'Other': 'Investigate the candidate details further and provide a customized recovery plan based on the latest call context.'
                }
                return mapping.get(reason_label, mapping['Other'])
 
         # Determine primary inferred reason based on selected checkboxes
            inferred_reason = 'No Notes Provided'
            if joined_competitor: inferred_reason = 'Joined another institution'
            elif already_working: inferred_reason = 'Already Working/Internship'
            elif looking_for_job_internship: inferred_reason = 'Looking for Job/Internship'
            elif not_interested: inferred_reason = 'Not Interested'
            elif financial_issue: inferred_reason = 'Financial issues'
            elif unreachable_not_connected: inferred_reason = 'Unreachable/Not Connected'
            elif decision_pending: inferred_reason = 'Decision Pending/Discussing'
 
            recommended_action = heuristic_recommendation(inferred_reason)
 
            st.info(f"**Inferred Reason**: {inferred_reason}")
            st.warning(f"**Recommended Action**: {recommended_action}")
 
    else:
        st.warning("Model artifacts not loaded. Please ensure the notebook has been run to train and save the model.")



#  pages/4_📊_Model_Performance_Analysis.py
 
def page_model_performance():
    st.markdown("""
    <div class="page-header">
        <h1><i class="fa-solid fa-chart-line"></i> Model Performance</h1>
        <p>Evaluation metrics, feature importance, and model comparison for the churn prediction model.</p>
    </div>
    """, unsafe_allow_html=True)
 
    @st.cache_data
    def load_performance_data():
        try:
            model_eval_results = pd.read_csv(os.path.join(ARTIFACTS_PATH, 'model_evaluation_results.csv'))
            feature_importance_report = pd.read_csv(os.path.join(ARTIFACTS_PATH, 'feature_importance_report.csv'))
            return model_eval_results, feature_importance_report
        except FileNotFoundError:
            st.error("Model performance files not found. Please run the notebook to generate them.")
            return pd.DataFrame(), pd.DataFrame()
 
    model_eval_results_df, feature_importance_df = load_performance_data()
 
    if not model_eval_results_df.empty:
        st.subheader("Model Comparison")
        st.write("Here you can see a comparison of different models evaluated during training.")
        st.dataframe(model_eval_results_df.set_index('Model'))
 
        st.subheader("Best Model Performance Summary")
     # Assuming the first row of model_eval_results_df is the best tuned model or you have a way to identify it
     # For this demo, let's assume the best model name is 'LogisticRegression' (from notebook output)
        best_model_name = 'Logistic Regression'
        best_model_performance = model_eval_results_df[model_eval_results_df['Model'] == best_model_name].iloc[0]
 
        if not best_model_performance.empty:
            st.markdown(f"##### Performance of {best_model_name}")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("Accuracy", f"{best_model_performance['Accuracy']:.3f}")
            with col2: st.metric("Precision", f"{best_model_performance['Precision']:.3f}")
            with col3: st.metric("Recall", f"{best_model_performance['Recall']:.3f}")
            with col4: st.metric("F1 Score", f"{best_model_performance['F1 Score']:.3f}")
            with col5: st.metric("ROC-AUC", f"{best_model_performance['ROC-AUC']:.3f}")
        else:
            st.warning("Could not find performance data for the best model.")
 
 
        st.subheader("Feature Importance")
        st.write("This chart shows the most important features contributing to the churn prediction model.")
        try:
           st.image(os.path.join(ARTIFACTS_PATH, 'feature_importance.png'))
        except FileNotFoundError:
           st.warning("Feature importance plot not found. Please run the notebook to generate it.")
 
        st.subheader("Receiver Operating Characteristic (ROC) Curve")
        st.write("The ROC curve illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.")
        try:
           st.image(os.path.join(ARTIFACTS_PATH, 'roc_curve.png'))
        except FileNotFoundError:
           st.warning("ROC curve plot not found. Please run the notebook to generate it.")
 
        st.subheader("Churn Distribution")
        st.write("This plot shows the distribution of churned vs. non-churned candidates in the dataset.")
        try:
           st.image(os.path.join(ARTIFACTS_PATH, 'churn_distribution.png'))
        except FileNotFoundError:
           st.warning("Churn distribution plot not found. Please run the notebook to generate it.")
 
    else:
         st.info("No model performance data available to display. Please ensure data processing and model training are complete in the notebook.")


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


    if   "Overview"            in page: page_overview()
    elif "Notes"            in page: page_note_analysis()
    elif "Predictor"           in page: page_live_predictor()
    elif "Model Performance"   in page: page_model_performance()


if __name__ == "__main__":
    main()