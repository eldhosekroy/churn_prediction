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
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense AI – Candidate Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS – Premium Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #141428 50%, #0d1b2a 100%);
    }

    /* Hide default streamlit header */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(99,102,241,0.3);
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 8px;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(99,102,241,0.25);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #e2e8f0;
        font-size: 38px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 6px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #64748b;
    }
    .kpi-icon { font-size: 24px; margin-bottom: 8px; }
    .kpi-red .kpi-value { color: #f87171; }
    .kpi-green .kpi-value { color: #34d399; }
    .kpi-blue .kpi-value { color: #60a5fa; }
    .kpi-amber .kpi-value { color: #fbbf24; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(99,102,241,0.2) 0%, transparent 100%);
        border-left: 4px solid #6366f1;
        padding: 12px 20px;
        border-radius: 0 12px 12px 0;
        margin: 24px 0 16px 0;
    }
    .section-header h2 {
        color: #e2e8f0;
        font-size: 20px;
        font-weight: 700;
        margin: 0;
    }

    /* Page header */
    .page-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(139,92,246,0.15) 50%, rgba(6,182,212,0.1) 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .page-header h1 {
        color: #fff;
        font-size: 28px;
        font-weight: 800;
        margin: 0 0 6px 0;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .page-header p {
        color: #94a3b8;
        margin: 0;
        font-size: 14px;
    }

    /* Risk badges */
    .badge-high { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.4); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-medium { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-low { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid rgba(52,211,153,0.4); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }

    /* Candidate card */
    .candidate-card {
        background: rgba(30,30,60,0.7);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,12,41,0.8);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        border-radius: 10px;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #fff !important;
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
        background: rgba(99,102,241,0.2);
        color: #a78bfa;
    }

    /* Input fields */
    .stSelectbox > div, .stNumberInput > div, .stTextInput > div {
        background: rgba(30,30,60,0.7) !important;
        border-color: rgba(99,102,241,0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* Metric delta */
    [data-testid="stMetricDelta"] { font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# (Replicates model.py logic — files untouched)
# ─────────────────────────────────────────────

DATA_DIR = "./"

@st.cache_data
def load_data():
    candidate_profile = pd.read_csv(DATA_DIR + "Candidate Profile.csv")
    call_log          = pd.read_csv(DATA_DIR + "Call log.csv")
    executive_profile = pd.read_csv(DATA_DIR + "Executive Profile.csv")
    return candidate_profile, call_log, executive_profile


@st.cache_data
def load_churn_reasons():
    """Attempt to load churn reason outputs saved by model.py. Returns full and short dataframes or (None, None)."""
    full_path = os.path.join(DATA_DIR, 'candidates_with_suggested_reasons.csv')
    short_path = os.path.join(DATA_DIR, 'churn_reasons.csv')
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
            'has_no_response':         int(sub.str.contains('no response|no pickup|unreachable|voicemail').any()),
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
def load_model():
    try:
        with open(DATA_DIR + "churn_prediction_model.pkl", "rb") as f:
            data = pickle.load(f)
            if 'model_name' not in data and 'model' in data:
                data['model_name'] = data['model'].__class__.__name__
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
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <div style="font-size:40px;">🎯</div>
            <div style="font-size:18px; font-weight:800; color:#e2e8f0; margin-top:8px;">ChurnSense AI</div>
            <div style="font-size:11px; color:#64748b; margin-top:4px;">Candidate Analytics Platform</div>
        </div>
        <hr style="border-color:rgba(99,102,241,0.2); margin: 12px 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["📊 Overview",
             "🔍 Candidate Explorer",
             "📞 Call Log Analysis",
             "💰 Payment Analysis",
             "🤖 Live Predictor",
             "📈 Model Performance"],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color:rgba(99,102,241,0.2);margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:10px;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;margin:0 0 8px 0;'>Data Sources</p>", unsafe_allow_html=True)

        _src = [
            ("📋", "Candidate Profile", ".csv &nbsp;·&nbsp; 50 rows"),
            ("📞", "Call Log",           ".csv &nbsp;·&nbsp; 124 rows"),
            ("👔", "Executive Profile",  ".csv &nbsp;·&nbsp; 10 rows"),
            ("🤖", "Churn Model",        ".pkl &nbsp;·&nbsp; Saved Model"),
        ]
        for _icon, _name, _meta in _src:
            st.markdown(f"""
<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.18);
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
  <span style="font-size:13px;vertical-align:middle;">&#128737;&#65039;</span>
  <span style="font-size:10px;color:#64748b;vertical-align:middle;margin-left:6px;">
    <b style="color:#34d399;">Read-only</b> &mdash; no team files modified
  </span>
</div>""", unsafe_allow_html=True)


    return page


# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
def page_overview(df, call_log_proc, churn_full=None):
    st.markdown("""
    <div class="page-header">
        <h1>📊 Executive Overview</h1>
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
            <div class="kpi-icon">👥</div>
            <div class="kpi-title">Total Candidates</div>
            <div class="kpi-value kpi-blue" style="color:#60a5fa">{total}</div>
            <div class="kpi-sub">Enrolled in system</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">✅</div>
            <div class="kpi-title">Active</div>
            <div class="kpi-value" style="color:#34d399">{active}</div>
            <div class="kpi-sub">Training joined</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-title">Churned</div>
            <div class="kpi-value" style="color:#f87171">{int(churned)}</div>
            <div class="kpi-sub">Did not join training</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📉</div>
            <div class="kpi-title">Churn Rate</div>
            <div class="kpi-value" style="color:#fbbf24">{churn_rate:.1f}%</div>
            <div class="kpi-sub">Of all candidates</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📞</div>
            <div class="kpi-title">Total Calls</div>
            <div class="kpi-value" style="color:#a78bfa">{total_calls}</div>
            <div class="kpi-sub">Across all candidates</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Suggested Churn Reasons (from model outputs) ─────────
    if churn_full is not None and 'Suggested_Churn_Reason' in churn_full.columns:
        try:
            reasons = churn_full[churn_full['Churn'] == 1]['Suggested_Churn_Reason'].value_counts().reset_index()
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
        <h1>🔍 Candidate Explorer</h1>
        <p>Browse, filter, and inspect individual candidate records with call history.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────
    with st.expander("🎛️ Filter Candidates", expanded=True):
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
    tbl['Churn'] = tbl['Churn'].map({0: '✅ Active', 1: '🔴 Churned'})
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

    churn_label = "🔴 CHURNED" if row['Churn'] == 1 else "✅ ACTIVE"
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
        <h1>📞 Call Log Analysis</h1>
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
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">📞</div>
            <div class="kpi-title">Total Calls</div>
            <div class="kpi-value" style="color:#60a5fa">{len(call_log_proc)}</div>
            <div class="kpi-sub">Across all candidates</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">⏱️</div>
            <div class="kpi-title">Avg Duration</div>
            <div class="kpi-value" style="color:#a78bfa">{avg_dur:.1f} min</div>
            <div class="kpi-sub">Per call</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">🔄</div>
            <div class="kpi-title">Avg Calls/Candidate</div>
            <div class="kpi-value" style="color:#34d399">{avg_calls_per_cand:.1f}</div>
            <div class="kpi-sub">Follow-up rate</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">📊</div>
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
    st.markdown('<div class="section-header"><h2>📝 Call Remark Sentiment Keywords</h2></div>', unsafe_allow_html=True)

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
            ['🟢 Interested', '🔴 No Response', '💰 Payment Talk', '💡 Technical Talk'],
            sentiment_data['Active'], sentiment_data['Churned']
        ):
            st.markdown(f"""
            <div style="background:rgba(30,30,60,0.5); border:1px solid rgba(99,102,241,0.15);
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
    st.markdown('<div class="section-header"><h2>📅 Call Activity Timeline</h2></div>', unsafe_allow_html=True)
    timeline = call_log_proc.groupby(call_log_proc['Call_Date'].dt.date).size().reset_index()
    timeline.columns = ['Date', 'Calls']
    timeline['Date'] = pd.to_datetime(timeline['Date'])
    fig4 = go.Figure(go.Scatter(
        x=timeline['Date'], y=timeline['Calls'],
        mode='lines+markers',
        line=dict(color='#6366f1', width=2.5),
        marker=dict(size=7, color='#a78bfa'),
        fill='tozeroy', fillcolor='rgba(99,102,241,0.1)',
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
        <h1>💰 Payment Analysis</h1>
        <p>Fee collection status, payment methods, and financial risk by churn segment.</p>
    </div>
    """, unsafe_allow_html=True)

    total_revenue    = df['Total_Amount'].sum()
    collected        = df['Paid_amount'].sum()
    outstanding      = df['Outstanding_Amount'].sum()
    collection_rate  = collected / total_revenue * 100 if total_revenue > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">💵</div>
            <div class="kpi-title">Total Expected</div>
            <div class="kpi-value" style="color:#60a5fa; font-size:26px;">₹{total_revenue/1e5:.2f}L</div>
            <div class="kpi-sub">Course fees total</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">✅</div>
            <div class="kpi-title">Collected</div>
            <div class="kpi-value" style="color:#34d399; font-size:26px;">₹{collected/1e5:.2f}L</div>
            <div class="kpi-sub">{collection_rate:.1f}% collected</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">⚠️</div>
            <div class="kpi-title">Outstanding</div>
            <div class="kpi-value" style="color:#f87171; font-size:26px;">₹{outstanding/1e5:.2f}L</div>
            <div class="kpi-sub">Pending recovery</div></div>""", unsafe_allow_html=True)
    with k4:
        churn_outstanding = df[df['Churn']==1]['Outstanding_Amount'].sum()
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">🚨</div>
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
        <h1>🤖 Live Churn Predictor</h1>
        <p>Fill in candidate details to get an instant AI-powered churn risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_data is None:
        st.error("⚠️ Could not load churn_prediction_model.pkl. Please ensure the model file is present in the project directory.")
        return

    model           = model_data['model']
    scaler          = model_data['scaler']
    feature_columns = model_data['feature_columns']
    label_encoders  = model_data['label_encoders']
    categorical_feat= model_data['categorical_features']
    numerical_feat  = model_data['numerical_features']

    st.markdown('<div class="section-header"><h2>Candidate Details</h2></div>', unsafe_allow_html=True)

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
        total_calls     = st.number_input("Total Calls", 0, 5, 1, key="p_tc")
        unique_execs    = st.number_input("Unique Executives", 0, 3, 1, key="p_ue")
    with cc2:
        total_call_dur  = st.number_input("Total Call Duration (min)", 0.0, 30.0, 8.0, step=0.5, key="p_tcd")
        avg_call_dur    = st.number_input("Avg Call Duration (min)",   0.0,  10.0, 4.0, step=0.5, key="p_acd")
    with cc3:
        max_call_dur    = st.number_input("Max Call Duration (min)",   0.0,  15.0, 6.0, step=0.5, key="p_mxcd")
        min_call_dur    = st.number_input("Min Call Duration (min)",   0.0,   8.0, 2.0, step=0.5, key="p_mncd")
    with cc4:
        call_freq       = st.number_input("Call Frequency (per month)", 0.0, 0.5, 0.18, step=0.01, key="p_cf")
        exec_exp        = st.number_input("Avg Executive Experience (yrs)", 0.0, 10.0, 5.0, step=0.5, key="p_ee")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: has_interest    = st.checkbox("Showed Interest in Calls", value=True, key="p_hi")
    with sc2: has_no_resp     = st.checkbox("No Response / Unreachable", value=False, key="p_hnr")
    with sc3: has_payment     = st.checkbox("Payment Discussion in Calls", value=False, key="p_hpd")
    with sc4: has_technical   = st.checkbox("Technical Discussion", value=True, key="p_htd")

    # Free-text call remarks for live inference
    call_remarks = st.text_area("Call Remarks (optional)", value="", max_chars=1000, placeholder="Enter recent call remarks or notes...", key="p_remarks")

    exec_team = st.selectbox("Executive Team", sorted(df['Executive_Team'].dropna().unique()) if 'Executive_Team' in df.columns else ['Team A','Team B','Team C','Team D'], key="p_et")

    days_since_payment = st.number_input("Days Since Last Payment", 0, 1000, 60, key="p_dsp")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary"):
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
                    <div class="pred-label" style="color:#f87171;">🔴</div>
                    <div style="font-size:24px; font-weight:800; color:#f87171; margin-bottom:8px;">HIGH CHURN RISK</div>
                    <div class="pred-sub">This candidate is likely to NOT join training</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-safe">
                    <div class="pred-label" style="color:#34d399;">✅</div>
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
            risk_level = "🔴 High" if prob > 0.6 else ("🟡 Medium" if prob > 0.35 else "🟢 Low")
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
                <hr style="border-color:rgba(99,102,241,0.2); margin:12px 0;">
                <div style="font-size:12px; color:#64748b;">
                    {'⚠️ <b style="color:#fbbf24;">Action Required:</b> Schedule immediate follow-up call.' if pred==1 else '✅ <b style="color:#34d399;">On Track:</b> Continue regular follow-up.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Use the same suggestion logic as model.py for parity
        def suggest_reason_from_text(remarks_text, feedback_text):
            text = ''
            if remarks_text:
                text += str(remarks_text).lower() + ' '
            if feedback_text:
                text += str(feedback_text).lower()

            # Priority-based keyword matching (same rules as model.py)
            if any(k in text for k in ['pay', 'payment', 'fee', 'installment', 'emi', 'finance', 'financial']):
                return 'Financial issues'
            if any(k in text for k in ['not interested', 'no interest', 'lack of interest', 'lost interest', 'not keen', 'disinterested', 'no longer interested']):
                return 'Lack of interest'
            if any(k in text for k in ['joined another', 'joined other', 'admission elsewhere', 'admitted', 'migrated to', 'joined institute', 'joined company', 'enrolled elsewhere']):
                return 'Joined another institution'
            if any(k in text for k in ['no response', 'no pickup', 'unreachable', 'voicemail', 'did not pick', 'not reachable', 'no answer', 'call dropped', 'busy', 'no contact', 'not responding']):
                return 'Communication gaps'

            # Fallbacks based on short signals
            if any(k in text for k in ['course not suitable', 'course mismatch', 'course not for me', 'content not relevant']):
                return 'Lack of interest'

            # Also consider payment flags and call flags heuristically
            if has_payment or (payment_ratio is not None and payment_ratio < 0.5):
                return 'Financial issues'
            if has_no_resp:
                return 'Communication gaps'

            return 'Other'

        suggested_reason = suggest_reason_from_text(call_remarks, feedback)

        st.markdown(f"<div style='margin-top:12px; padding:12px; border-radius:8px; background:rgba(99,102,241,0.06);'>"
                    f"<b>Suggested Churn Reason:</b> {suggested_reason}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 6 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
def page_model_performance(df, model_data):
    st.markdown("""
    <div class="page-header">
        <h1>📈 Model Performance</h1>
        <p>Evaluation metrics, feature importance, and model comparison for the churn prediction model.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_data is None:
        st.error("⚠️ Model file not found.")
        return

    model           = model_data['model']
    feature_columns = model_data['feature_columns']
    balance_method  = model_data.get('balance_method', 'None')
    balance_label   = format_balance_method(balance_method)
    balance_note    = balance_method_description(balance_method)
    model_name      = model_data.get('model_name') or model.__class__.__name__
    friendly_model  = {
        'RandomForestClassifier': 'Random Forest',
        'GradientBoostingClassifier': 'Gradient Boosting',
        'XGBClassifier': 'XGBoost',
        'LogisticRegression': 'Logistic Regression'
    }.get(model_name, model_name)

    # ── Model Info ────────────────────────────────
    st.markdown('<div class="section-header"><h2>Model Information</h2></div>', unsafe_allow_html=True)
    i1, i2, i3, i4, i5 = st.columns(5)
    with i1:
        st.markdown(f"""<div class=\"kpi-card\"><div class=\"kpi-icon\">🤖</div>
            <div class=\"kpi-title\">Algorithm</div>
            <div class=\"kpi-value\" style=\"color:#a78bfa; font-size:16px; margin-top:8px;\">{friendly_model}</div>
            <div class=\"kpi-sub\">Deployed model</div></div>""", unsafe_allow_html=True)
    with i2:
        estimator_value = getattr(model,'n_estimators', None) or getattr(model,'n_estimators_', 'N/A')
        st.markdown(f"""<div class=\"kpi-card\"><div class=\"kpi-icon\">🌳</div>
            <div class=\"kpi-title\">Estimators</div>
            <div class=\"kpi-value\" style=\"color:#60a5fa;\">{estimator_value}</div>
            <div class=\"kpi-sub\">Decision trees</div></div>""", unsafe_allow_html=True)
    with i3:
        max_depth_value = getattr(model,'max_depth', 'N/A')
        st.markdown(f"""<div class=\"kpi-card\"><div class=\"kpi-icon\">📐</div>
            <div class=\"kpi-title\">Max Depth</div>
            <div class=\"kpi-value\" style=\"color:#34d399;\">{max_depth_value}</div>
            <div class=\"kpi-sub\">Tree depth limit</div></div>""", unsafe_allow_html=True)
    with i4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">🔢</div>
            <div class="kpi-title">Features Used</div>
            <div class="kpi-value" style="color:#fbbf24;">{len(feature_columns)}</div>
            <div class="kpi-sub">Input dimensions</div></div>""", unsafe_allow_html=True)
    with i5:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">⚖️</div>
            <div class="kpi-title">Balancing</div>
            <div class="kpi-value" style="color:#fbbf24; font-size:16px; margin-top:8px;">{balance_label}</div>
            <div class="kpi-sub">Imbalance handling</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>Balancing Technique</h2></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="candidate-card">
        <div style="display:flex; justify-content:space-between; gap:16px; align-items:center;">
            <div>
                <div style="font-size:13px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Selected Method</div>
                <div style="font-size:24px; font-weight:800; color:#fbbf24; margin-top:4px;">{balance_label}</div>
            </div>
            <div style="max-width:620px; color:#94a3b8; font-size:13px; line-height:1.5;">{balance_note}</div>
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
                    <div style="background:rgba(99,102,241,0.1); border-radius:4px; height:5px;">
                        <div style="width:{pct:.0f}%; background:linear-gradient(90deg,#6366f1,#8b5cf6); height:100%; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Churn Reason Analysis ─────────────────────
    st.markdown('<div class="section-header"><h2>📋 Why Are Candidates Churning? — Reason Analysis</h2></div>', unsafe_allow_html=True)

    churned_df = df[df['Churn'] == 1].copy()
    active_df  = df[df['Churn'] == 0].copy()

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown("""
        <div class="candidate-card">
            <div style="font-size:13px; font-weight:700; color:#f87171; margin-bottom:12px;">🔴 Zero Payment</div>
        """, unsafe_allow_html=True)
        zero_pay = (churned_df['Paid_amount'] == 0).sum()
        pct = zero_pay / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
            <div style="font-size:32px; font-weight:800; color:#f87171;">{zero_pay}</div>
            <div style="color:#64748b; font-size:13px;">churned with ₹0 paid ({pct:.0f}%)</div>
            <div style="margin-top:10px; font-size:12px; color:#475569;">Most churned candidates made no payment at all — a strong churn signal.</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div class="candidate-card">
            <div style="font-size:13px; font-weight:700; color:#fbbf24; margin-bottom:12px;">📵 No Response Pattern</div>
        """, unsafe_allow_html=True)
        no_resp = churned_df.get('has_no_response', pd.Series([0]*len(churned_df))).sum() if 'has_no_response' in churned_df.columns else 0
        pct2 = no_resp / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
            <div style="font-size:32px; font-weight:800; color:#fbbf24;">{int(no_resp)}</div>
            <div style="color:#64748b; font-size:13px;">churned had no-response calls ({pct2:.0f}%)</div>
            <div style="margin-top:10px; font-size:12px; color:#475569;">Unreachable candidates rarely convert — early escalation is critical.</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown("""
        <div class="candidate-card">
            <div style="font-size:13px; font-weight:700; color:#a78bfa; margin-bottom:12px;">🙅 Not Attended Induction</div>
        """, unsafe_allow_html=True)
        not_attended = (churned_df['Induction_Session'] == 'NotAttended').sum() if 'Induction_Session' in churned_df.columns else 0
        pct3 = not_attended / len(churned_df) * 100 if len(churned_df) > 0 else 0
        st.markdown(f"""
            <div style="font-size:32px; font-weight:800; color:#a78bfa;">{not_attended}</div>
            <div style="color:#64748b; font-size:13px;">churned skipped induction ({pct3:.0f}%)</div>
            <div style="margin-top:10px; font-size:12px; color:#475569;">Skipping the induction session is a key early warning sign for churn.</div>
        </div>""", unsafe_allow_html=True)

    # ── Model Comparison Table (from model.py's logic) ──
    st.markdown('<div class="section-header"><h2>🏆 Algorithm Comparison (Reference)</h2></div>', unsafe_allow_html=True)

    comparison_data = {
        'Model':     ['Random Forest (Regularized)', 'Gradient Boosting (Reg)',
                      'XGBoost (Reg)', 'Random Forest', 'Gradient Boosting',
                      'XGBoost', 'Logistic Regression', 'AdaBoost', 'Decision Tree', 'SVM', 'KNN', 'Naive Bayes'],
        'Note':      ['✅ Selected', 'Tuned', 'Tuned', 'Baseline', 'Baseline',
                      'Baseline', 'Baseline', 'Baseline', 'Baseline', 'Baseline', 'Baseline', 'Baseline'],
        'Strength':  ['Balanced accuracy, low overfit', 'High F1, slight overfit',
                      'High accuracy', 'Good recall', 'High precision',
                      'Fast', 'Interpretable', 'Ensemble', 'Fast', 'High precision', 'Simple', 'Probabilistic'],
    }
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Model":    st.column_config.TextColumn("Algorithm"),
                     "Note":     st.column_config.TextColumn("Status"),
                     "Strength": st.column_config.TextColumn("Characteristic"),
                 })


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    page = sidebar()

    # Load data
    with st.spinner("Loading data..."):
        try:
            candidate_profile, call_log, executive_profile = load_data()
            df, call_log_proc = preprocess(candidate_profile, call_log, executive_profile)
            churn_full, churn_short = load_churn_reasons()
        except FileNotFoundError as e:
            st.error(f"⚠️ Could not load data files: {e}\n\nPlease ensure the CSV files are in the same directory as dashboard.py.")
            st.stop()

    model_data = load_model()

    if   "Overview"            in page: page_overview(df, call_log_proc, churn_full)
    elif "Candidate Explorer"  in page: page_candidate_explorer(df, call_log_proc, executive_profile, churn_full)
    elif "Call Log"            in page: page_call_analysis(df, call_log_proc, executive_profile)
    elif "Payment"             in page: page_payment_analysis(df)
    elif "Predictor"           in page: page_live_predictor(df, model_data)
    elif "Model Performance"   in page: page_model_performance(df, model_data)


if __name__ == "__main__":
    main()
