import os

import pandas as pd
import numpy as np

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

supabase = get_supabase_client()

def import_crm_leads_pipeline(file_path: str):
    """Parses incoming spreadsheets and maps columns directly into the candidates table."""
    try:
        print(f" Processing source data file: {file_path}")
        df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

        # Structure Translation Map
        mapping_dict = {
            'Contact Name': 'candidate_name', 'Contact Id': 'contact_id',
            'Email ID': 'email', 'Whatsapp Number': 'contact_phone',
            'Gender': 'gender', 'Education': 'education', 'Stream': 'stream',
            'City': 'city', 'Mailing State': 'mailing_state', 'Mailing Country': 'mailing_country',
            'Course': 'course', 'Track Interested': 'track_interested', 'Batch Assigned': 'batch_assigned',
            'Mode of Program Joined': 'program_mode', 'Program Location': 'program_location',
            'Induction Session': 'induction_session', 'Feedback': 'interest_level', 'final_inferred_reason': 'background_override'
        }
        df = df.rename(columns={k: v for k, v in mapping_dict.items() if k in df.columns})
        df = df.dropna(subset=['email', 'candidate_name']).replace({np.nan: None})

        # Sanitize interest level tags
        if 'interest_level' in df.columns:
            df['interest_level'] = df['interest_level'].astype(str).str.lower().str.strip()
            df['interest_level'] = df['interest_level'].apply(
                lambda x: x if x in ['high', 'medium', 'low'] else 'medium')

        schema_cols = [
            'candidate_name', 'contact_id', 'email', 'contact_phone', 'gender',
            'education', 'stream', 'background_override', 'city', 'mailing_state',
            'mailing_country', 'course', 'track_interested', 'batch_assigned',
            'program_mode', 'program_location', 'induction_session', 'interest_level'
        ]
        records = df[[c for c in df.columns if c in schema_cols]].to_dict(orient='records')

        # Run safe database upsert
        supabase.table("candidates").upsert(records, on_conflict="contact_id").execute()
        print(f" Lead Engine complete! Uploaded {len(records)} entries successfully.")
        return True
    except Exception as e:
        print(f" Ingestion failed: {e}")
        return False


if __name__ == "__main__":
    file = os.path.join(os.path.dirname(__file__), "output", "enrolled_processed.csv")
    import_crm_leads_pipeline(file)