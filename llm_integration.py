import os
import json
import requests
import re
from dotenv import load_dotenv

load_dotenv()

def generate_prompt(candidate_data):
    return f"""
You are an expert HR Retention Analyst. Analyze this candidate's profile and predict their likelihood of churning (dropping out / requesting a refund).

Candidate Details:
- Experience: {candidate_data.get('Experience')}
- Semester: {candidate_data.get('Semester')}
- Year of Graduation: {candidate_data.get('Year of Graduation')}
- Education: {candidate_data.get('Course')}
- Source of Lead: {candidate_data.get('Source of lead')}
- Batch Assigned: {candidate_data.get('Batch Assigned to')}
- Track Interested: {candidate_data.get('Track Interested')}
- Gender: {candidate_data.get('Gender')}

Risk Flags:
- Has Financial Issues: {'Yes' if candidate_data.get('financial_issue') else 'No'}
- Not Interested: {'Yes' if candidate_data.get('not_interested') else 'No'}
- Already Working / Placed: {'Yes' if candidate_data.get('already_working') else 'No'}
- Decision Pending: {'Yes' if candidate_data.get('decision_pending') else 'No'}
- Unreachable / Not Connected: {'Yes' if candidate_data.get('unreachable_not_connected') else 'No'}
- Payment Done: {'Yes' if candidate_data.get('Invoice_binary') else 'No'}

Respond strictly in valid JSON format with exactly the following three keys:
{{
  "churn_probability": <A float between 0.0 and 100.0>,
  "reason": "<A concise 1-2 sentence explanation of why they are at risk or safe>",
  "retention_strategy": "<A highly actionable 1-2 sentence strategy on how to keep this specific candidate>"
}}
"""

def parse_json_response(text):
    text = text.strip()
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    
    try:
        data = json.loads(text)
        return data
    except Exception as e:
        return {
            "churn_probability": 50.0,
            "reason": f"Failed to parse LLM response. Raw output: {text[:100]}...",
            "retention_strategy": "Please try again or select a different AI model."
        }

def predict_with_gemini(candidate_data):
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env")
    genai.configure(api_key=api_key)
    # Using 1.5-flash as 2.5 might not be available in older library version
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = generate_prompt(candidate_data)
    response = model.generate_content(prompt)
    return parse_json_response(response.text)

def predict_with_groq(candidate_data):
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env")
    client = Groq(api_key=api_key)
    prompt = generate_prompt(candidate_data)
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"}
    )
    return json.loads(chat_completion.choices[0].message.content)

def predict_with_hf(candidate_data):
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY is not set in .env")
    
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = generate_prompt(candidate_data)
    
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {"max_new_tokens": 250, "temperature": 0.1, "return_full_text": False}
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise ValueError(f"Hugging Face API Error: {response.text}")
        
    result_text = response.json()[0]['generated_text'].strip()
    return parse_json_response(result_text)

def get_llm_prediction(model_name, candidate_data):
    if "Gemini" in model_name:
        return predict_with_gemini(candidate_data)
    elif "Groq" in model_name:
        return predict_with_groq(candidate_data)
    elif "Hugging Face" in model_name:
        return predict_with_hf(candidate_data)
    else:
        raise ValueError(f"Unknown LLM Model Selected: {model_name}")
