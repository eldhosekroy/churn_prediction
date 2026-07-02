📞 Call Analyzer Pro
AI-Powered Call Analysis & Transcription Tool built with Streamlit

🎯 Overview
Call Analyzer Pro is a comprehensive web application that analyzes call recordings and transcriptions to extract meaningful insights, generate call remarks, and categorize calls automatically.

✨ Features
📤 Audio Upload - Upload call recordings in multiple formats (MP3, WAV, M4A, OGG, FLAC, WebM)
📝 Text Transcription - Paste call transcriptions directly
🌐 Language Detection - Automatically detect Malayalam or English
🔄 Translation - Translate Malayalam transcriptions to English
📊 Call Duration - Extract call duration from audio files
🏷️ Call Remarks - Generate concise 2-3 word conclusions
🔑 Keyword Extraction - Extract key topics and entities
📈 Sentiment Analysis - Determine call sentiment (positive/negative/neutral)
🏷️ Call Categorization - Classify calls as Sales, Support, Inquiry, etc.
📁 Project Structure
sh

Copy
call_analyzer/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration and settings
├── audio_processor.py     # Audio transcription and processing
├── text_processor.py      # Text transcription processing
├── translator.py          # Malayalam to English translation
├── keyword_extractor.py   # Call remarks and keyword extraction
├── requirements.txt       # Python dependencies
└── README.md              # This file
🚀 Quick Start
Prerequisites
Python 3.8+
pip (Python package installer)
Installation
Clone or download the project

sh

Copy
cd call_analyzer
Install dependencies

sh

Copy
pip install -r requirements.txt
Run the application

sh

Copy
streamlit run app.py
Open in browser The app will automatically open at http://localhost:8501

🏗️ Architecture
Module Overview
sh

Copy
┌─────────────────────────────────────────────────────────────┐
│                        app.py                               │
│                  (Main Streamlit App)                       │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │   Audio   │      │   Text    │      │ Translator│
    │ Processor │      │ Processor │      │           │
    └───────────┘      └───────────┘      └───────────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Keyword Extractor│
                    │  (Call Remarks)  │
                    └─────────────────┘
1. config.py - Configuration Module
Contains all application settings:

Supported languages
Audio file settings
Analysis parameters
UI customization
2. audio_processor.py - Audio Processing
Responsibilities:

Validate audio files
Extract call duration
Transcribe audio to text
Key Class: AudioProcessor

Methods:

validate_audio_file() - Check file format and size
get_audio_duration() - Extract duration from audio
transcribe_audio() - Convert audio to text (placeholder for API)
3. text_processor.py - Text Processing
Responsibilities:

Detect language (Malayalam/English)
Preprocess and clean text
Analyze text structure
Extract speaker segments
Key Class: TextProcessor

Methods:

detect_language() - Identify language from text
preprocess_text() - Clean and normalize text
split_into_sentences() - Tokenize text
analyze_text_structure() - Get word/sentence statistics
4. translator.py - Translation Module
Responsibilities:

Translate Malayalam text to English
Handle multiple language pairs
Key Class: Translator

Methods:

translate() - General translation function
translate_malayalam_to_english() - ML to EN translation
batch_translate() - Translate multiple texts
5. keyword_extractor.py - Keyword & Remarks Extraction
Responsibilities:

Extract important keywords
Generate call remarks (2-3 words)
Categorize calls
Extract entities (order IDs, phone numbers, etc.)
Analyze sentiment
Key Class: KeywordExtractor

Methods:

extract_keywords() - Get top keywords from text
extract_entities() - Find order IDs, phone numbers, etc.
generate_call_remarks() - Create concise summary
extract_call_summary() - Complete summary generation
6. app.py - Main Application
Structure:

Header - App branding and title
Sidebar - Settings and configuration
Tab Interface - Audio upload / Text input
Results Display - Comprehensive analysis output
Key Functions:

render_header() - Display app header
render_sidebar() - Settings panel
render_audio_upload() - File upload interface
render_text_input() - Text input area
display_analysis_results() - Show results with metrics
process_audio() - Audio analysis pipeline
process_text() - Text analysis pipeline
🔧 Configuration
Audio Settings (config.py)
sh

Copy
AUDIO_SETTINGS = {
    "max_file_size_mb": 50,
    "supported_formats": ["mp3", "wav", "m4a", "ogg", "flac", "webm"],
    "sample_rate": 16000,
}
Call Categories
sh

Copy
CALL_CATEGORIES = {
    "sales": ["sale", "purchase", "buy", "price", "offer", ...],
    "support": ["help", "issue", "problem", "support", ...],
    "inquiry": ["information", "details", "query", ...],
    "scheduling": ["appointment", "schedule", "meeting", ...],
    "followup": ["follow", "callback", "reminder", ...],
    "general": [],
}
🎨 UI Components
Main Sections
Header - Gradient styled title
Sidebar - Settings and mode selection
Tab Interface - Audio or Text input
Analysis Results:
Overview Metrics (Duration, Language, Category, Sentiment)
Transcription Display
Call Remarks (highlighted prominently)
Key Topics
Keyword Analysis Table
Extracted Entities
🔌 Integration Guide
Adding Real Transcription (OpenAI Whisper)
sh

Copy
# In audio_processor.py, update transcribe_audio method:

def transcribe_audio(self, file, language: str = "auto"):
    import openai
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    
    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
                language=language if language != "auto" else None
            )
        return {
            "success": True,
            "text": transcript["text"],
            "language": language,
        }
    finally:
        os.remove(tmp_path)
Adding Real Translation (DeepL)
sh

Copy
# In translator.py, update translate_malayalam_to_english:

def translate_malayalam_to_english(self, malayalam_text: str):
    import deepl
    
    translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))
    result = translator.translate_text(
        malayalam_text,
        source_lang="ML",
        target_lang="EN"
    )
    
    return {
        "success": True,
        "translated_text": result.text,
        "source_language": "malayalam",
        "target_language": "english",
    }
📱 Sample Workflows
Workflow 1: English Audio Analysis
sh

Copy
User uploads English audio → Audio transcribed (English) → 
Keywords extracted → Call remarks generated
Workflow 2: Malayalam Audio Analysis
sh

Copy
User uploads Malayalam audio → Audio transcribed (Malayalam) → 
Translated to English → Keywords extracted → Call remarks generated
Workflow 3: Malayalam Text Analysis
sh

Copy
User pastes Malayalam text → Language detected (Malayalam) → 
Translated to English → Keywords extracted → Call remarks generated
🎯 Output Examples
Sample Call Remarks
Input Type	Call Remarks
Sales Call	"Product Inquiry", "Pricing Discussion"
Support Call	"Support Request", "Issue Resolution"
Inquiry	"Information Request", "Details Query"
Sample Output Structure
sh

Copy
{
    "call_remarks": "Product Inquiry",
    "secondary_remarks": ["Pricing", "Order", "Delivery"],
    "category": "sales",
    "key_topics": ["product", "price", "order", "delivery", "available"],
    "sentiment": "neutral",
    "entities": {
        "order_ids": ["ORD-12345"],
        "phone_numbers": ["+91 9876543210"],
        "emails": [],
        "amounts": [],
        "dates": []
    },
    "summary": "Customer inquiring about product pricing and ordering process."
}
🛠️ Customization
Adding New Languages
Update SUPPORTED_LANGUAGES in config.py
Add language detection patterns in text_processor.py
Add translation support in translator.py
Adding New Call Categories
Update CALL_CATEGORIES in config.py
Add category keywords in keyword_extractor.py
Update remark templates if needed
Styling
Modify CSS in app.py render_header() function:

sh

Copy
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)
🐛 Troubleshooting
Common Issues
Import errors: Ensure all .py files are in the same directory
File too large: Check file size is under 50MB
Unsupported format: Use MP3, WAV, M4A, OGG, FLAC, or WebM
Demo Mode
Enable "Demo Mode" in sidebar to test without actual API integrations.

📄 License
MIT License - Feel free to use and modify.

🤝 Contributing
Fork the repository
Create a feature branch
Make your changes
Submit a pull request
Built using Streamlit