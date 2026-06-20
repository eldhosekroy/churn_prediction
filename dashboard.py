"""## **Streamlit App Creation**

### Install Streamlit
"""


"""### Create Streamlit App Structure"""
import numpy as np
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle



# Create the 'pages' directory if it doesn't exist
if not os.path.exists('pages'):
    os.makedirs('pages')

print("Created 'pages' directory.")



 

ARTIFACTS_PATH = './output'

 
st.set_page_config(
     page_title="Churn Prediction Dashboard",
     page_icon="📊",
     layout="wide"
 )
 
st.title("📊 Churn Prediction Dashboard")
st.markdown("--- ")
 
st.markdown(
     """
     Welcome to the Churn Prediction Dashboard! This interactive tool provides insights into candidate data,
     churn prediction, and model performance.
 
     Use the sidebar to navigate through different sections:
     - **Candidate Exploration**: Dive into the overall candidate data.
     - **Notes Analysis**: Understand the insights extracted from candidate notes.
     - **Candidate Churn Prediction**: Predict churn for new candidates and get actionable recommendations.
     - **Model Performance Analysis**: Evaluate the performance of the churn prediction model.
     """
 )
 
st.subheader("Project Overview")
st.write("This project aims to predict candidate churn and provide actionable insights to improve retention. It leverages data from CRM and notes to build a robust predictive model.")
 
st.subheader("Key Sections")
st.markdown(
     """
     - **📈 Candidate Exploration**: Visualizations of candidate demographics, program choices, and other key attributes.
     - **📝 Notes Analysis**: Text analysis of call notes to identify common reasons for joining or not joining.
     - **🔮 Candidate Churn Prediction**: An interactive tool to get live churn predictions and suggested interventions.
     - **📊 Model Performance Analysis**: Detailed metrics and visualizations to understand how well the model performs.
     """
 )

 
 
 
st.set_page_config(
     page_title="Candidate Exploration",
     page_icon="📈",
     layout="wide"
 )
 
st.title("📈 Candidate Exploration")
st.markdown("--- ")
 
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
 
st.set_page_config(
     page_title="Notes Analysis",
     page_icon="📝",
     layout="wide"
)
 
st.title("📝 Notes Analysis")
st.markdown("--- ")
 
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

 
st.set_page_config(
     page_title="Churn Prediction",
     page_icon="🔮",
     layout="wide"
 )
 
st.title("🔮 Candidate Churn Prediction")
st.markdown("--- ")
 
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
 
 
st.set_page_config(
     page_title="Model Performance",
     page_icon="📊",
     layout="wide"
 )
 
st.title("📊 Model Performance Analysis")
st.markdown("--- ")
 
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