import streamlit as st
import re
import requests
import json
import os


def live_groq_pipeline(text, api_key):
    """
    Connects to Groq using a raw REST API call. Forces JSON Mode
    with upgraded 70B parameters to prevent token boundary truncation.
    """
    if not text.strip():
        return None

    is_malayalam = bool(re.search(r'[\u0D00-\u0D7F]', text))
    detected_lang = "Malayalam (Native Script Detected)" if is_malayalam else "English / Mixed Script"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Refined prompt instructing concise outputs to remain well within token headroom boundaries
    prompt = f"""
    You are an expert recruitment call analyzer. Analyze this transcript content: "{text}"

    Perform the following tasks concisely:
    1. Translate the text into clear English if it's in Malayalam. Give the entire translation.
    2. Determine the Sentiment (Positive, Negative, or Neutral).
    3. Extract 3 to 5 core English keywords representing structural churn focus areas (e.g., salary, commute, distance, offer, timing).
    4. Write a 1-2 sentence Summary indicating if the candidate will join or not, extracting the primary reason for churn if they decline.

    Return your output strictly in this JSON layout:
    {{
        "translation": "English translation goes here",
        "sentiment": "Positive, Negative, or Neutral",
        "keywords": ["keyword1", "keyword2"],
        "summary": "1-2 sentence final verdict summary goes here"
    }}
    """

    payload = {
        # FIX: Switched to Groq's active 70B production model
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system",
             "content": "You are a precise data extractor that outputs short, clean sentences strictly in JSON format."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens": 1000
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            parsed_json = json.loads(response.json()["choices"][0]["message"]["content"])
            parsed_json["lang"] = detected_lang
            return parsed_json
        else:
            # Fallback model strategy if specific regional rate limits hit the 70B tier
            if "model" in payload and payload["model"] == "llama-3.3-70b-specdec":
                payload["model"] = "llama3-70b-8192"
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    parsed_json = json.loads(response.json()["choices"][0]["message"]["content"])
                    parsed_json["lang"] = detected_lang
                    return parsed_json
            st.error(f"Groq Core Analysis Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Analysis Network Error: {str(e)}")
        return None


def transcribe_malayalam_audio(audio_file, api_key):
    """
    Transcribes sales call audio with absolute precision by embedding conversational
    dynamics, candidate objections, and mixed Malayalam/English speech patterns.
    """
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    temp_filename = f"temp_{audio_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(audio_file.read())

    # THE PERFECTED SALES CONTEXT PROMPT:
    # This acts as an active semantic guide, teaching Whisper to identify common
    # admission hurdles, direct rejections, and natural bilingual phrasing.
    perfect_sales_prompt = (
        "Salesperson is trying to admit the candidate into their training course. "
        "Some candidates may reject directly, some may show concern or tell the reason, "
        "some will be interested. Some say high fees so can't join, looking for a job, or not interested. "
        "The candidate is expressing it in Malayalam, mixing common words like: "
        "fees, high fees, salary, interview, course, training, job, internship, join, reject, not interested."
    )

    try:
        with open(temp_filename, "rb") as f_bytes:
            files = {
                "file": (temp_filename, f_bytes, audio_file.type)
            }
            data = {
                "model": "whisper-large-v3",
                "language": "ml",  # Locks primary phonetic decoding onto Malayalam
                "prompt": perfect_sales_prompt,  # Injects the complete domain blueprint
                "temperature": 0.0  # Hard-locks deterministic translation tracking
            }

            # --- PRIMARY TRANSCRIPTION PASS ---
            response = requests.post(url, headers=headers, files=files, data=data, timeout=60)

            # --- BACKUP FALLBACK PASS ---
            if response.status_code != 200:
                data["model"] = "whisper-large-v3-turbo"
                f_bytes.seek(0)
                response = requests.post(url, headers=headers, files=files, data=data, timeout=60)

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            st.error(f"Groq Audio Transcription Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        st.error(f"Audio Engine Connection Failed: {str(e)}")
        return None


# --- STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="Groq Call Intel Analyzer", layout="wide")

st.title(" Call Processing & Churn Analysis Intelligence")
st.markdown("Powered by Groq • End-to-End Malayalam Audio Transcription & Analysis Matrix")

# --- SIDEBAR AUTHENTICATION ---
st.sidebar.header(" Groq Credentials")
groq_key = st.sidebar.text_input("Enter Groq API Key", type="password", placeholder="gsk_...")

if groq_key:
    st.sidebar.success("Groq Routing Engine Online!")
else:
    st.sidebar.warning("Paste your free Groq key in the sidebar to activate the process hooks.")

tab1, tab2 = st.tabs([" Upload Audio Call", " Paste Transcribed Text"])

final_result = None
raw_transcript_preview = ""

# --- TAB 1: AUDIO CONFIGURATION ---
with tab1:
    st.subheader(" Upload Malayalam Recording")
    audio_file = st.file_uploader("Upload Audio Call Recording", type=["wav", "mp3", "m4a"])

    if audio_file:
        st.audio(audio_file)
        if st.button("Process Audio Call", key="process_audio"):
            if not groq_key:
                st.error("Please insert your Groq API key in the left sidebar first.")
            else:
                with st.spinner("Transcribing Malayalam Audio with Whisper-V3..."):
                    malayalam_text = transcribe_malayalam_audio(audio_file, groq_key)

                    if malayalam_text:
                        raw_transcript_preview = malayalam_text
                        with st.spinner("Translating text and running metrics pipelines..."):
                            final_result = live_groq_pipeline(malayalam_text, groq_key)

# --- TAB 2: TRANSCRIPT CONFIGURATION ---
with tab2:
    st.subheader(" Input Written Text Content")
    pasted_text = st.text_area("Paste text here:", height=150,
                               placeholder="ഞാൻ നാളെ വരാം, പക്ഷെ ഓഫീസ് വളരെ ദൂരെയാണ്...")

    if st.button("Analyze with Groq Engine", key="process_text"):
        if not groq_key:
            st.error("Please insert your Groq API key in the left sidebar first.")
        elif pasted_text.strip():
            raw_transcript_preview = pasted_text
            with st.spinner("Streaming data through JSON validation layers..."):
                final_result = live_groq_pipeline(pasted_text, groq_key)
        else:
            st.warning("Please fill the text input space before running execution loops.")

# --- PRESENT ANALYTICS RESULTS ---
if final_result:
    st.divider()
    st.subheader(" Operational Analytics Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=" Processing Mode",
                  value="Audio Input Pipeline" if "process_audio" in st.session_state or audio_file else "Text Input Pipeline")
    with col2:
        st.metric(label=" Identified Language", value=final_result.get("lang", "Unknown"))
    with col3:
        st.metric(label=" Candidate Sentiment", value=final_result.get("sentiment", "Neutral"))
    with col4:
        st.markdown("** Extracted Keywords (English Only):**")
        kws = final_result.get("keywords", [])
        st.write(", ".join([f"`{k}`" for k in kws]) if kws else "_None_")

    st.subheader(" 1-2 Sentence Summary (Used for Churn Extraction)")
    st.info(final_result.get("summary", "No summary could be processed."))

    with st.expander("️ View Live English Translation Generated by Llama"):
        st.write(final_result.get("translation", "No translation generated."))

    if raw_transcript_preview:
        with st.expander(" View Original Captured Malayalam Transcript"):
            st.write(raw_transcript_preview)