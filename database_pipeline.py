import os

import pandas as pd
import numpy as np

from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(override=True)

url: str = os.environ.get("SUPABASE_URL", "")
key: str = os.environ.get("SUPABASE_KEY", "")

# 1. INITIALIZE SUPABASE ROUTINES

def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def log_crm_call_interaction(
    email: str, duration_sec: int, agent_name: str, direction: str,
    transcript_text: str, remarks_text: str = None, remark_cat: str = None,
    outcomes_list: list = None, summary_text: str = None, interest: str = "medium",
    key_topics_list: list = None, followup_req: bool = False,
    next_followup_str: str = None, priority: str = None
):
    """Resolves transactional key bindings and writes communication history logs."""
    supabase = get_supabase_client()
    try:
        # A. Resolve mandatory candidate_id (UUID) via email tracking keys
        candidate_query = supabase.table("candidates").select("id").eq("email", email).limit(1).execute()
        if not candidate_query.data:
            print(f" Aborted: No candidate record matching email: {email}")
            return False
        candidate_uuid = candidate_query.data[0]["id"]

        # B. Grab latest prediction reference string if available
        prediction_uuid = None
        pred_query = supabase.table("predictions").select("id").eq("email", email).order("predicted_at", desc=True).limit(1).execute()
        if pred_query.data:
            prediction_uuid = pred_query.data[0]["id"]

        # C. Text analytics sentiment calculation algorithm (-1.00 to 1.00)
        if transcript_text:
            joy_words = ['interested', 'excited', 'yes', 'perfect', 'join', 'good', 'agree']
            sad_words = ['expensive', 'cancel', 'no', 'busy', 'unable', 'drop', 'bad']
            lower_text = transcript_text.lower()
            pos = sum(1 for w in joy_words if w in lower_text)
            neg = sum(1 for w in sad_words if w in lower_text)
            sentiment_score = round((pos - neg) / (pos + neg), 2) if (pos + neg) > 0 else 0.0
        else:
            sentiment_score = 0.0

        # D. Assemble call interaction configuration block
        call_payload = {
            "candidate_id": candidate_uuid,
            "prediction_id": prediction_uuid,
            "call_duration": int(duration_sec),
            "call_agent": agent_name,
            "call_direction": direction.lower().strip(),
            "outcomes": outcomes_list if outcomes_list else [],
            "remark_category": remark_cat if remark_cat else None,
            "remarks": remarks_text,
            "transcript": transcript_text,
            "transcript_summary": summary_text,
            "sentiment_score": float(sentiment_score),
            "interest_level": interest.lower().strip(),
            "key_topics": key_topics_list if key_topics_list else [],
            "followup_required": bool(followup_req),
            "next_followup_date": next_followup_str if followup_req else None,
            "followup_priority": priority.lower().strip() if (followup_req and priority) else None
        }

        supabase.table("calls").insert(call_payload).execute()
        return True
    except Exception as db_err:
        print(f" Call logging exception: {db_err}")
        return False