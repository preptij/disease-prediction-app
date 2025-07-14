import streamlit as st
import joblib
import os
import re
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

# ========== Paths ==========
MODEL_PATH = "notebooks/outputs/models/disease_model.joblib"
VECTORIZER_PATH = "notebooks/outputs/models/vectorizer.joblib"
DESC_PATH = "data/symptom_description.csv"
PRECAUTION_PATH = "data/symptom_precaution.csv"

# ========== Error Handling for Missing Files ==========
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("⚠️ Model or vectorizer file not found. Please train and save them in your notebook.")
    st.stop()

# ========== Load Trained Model and Vectorizer ==========
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ========== Load Support Data ==========
desc_df = pd.read_csv(DESC_PATH)
precautions_df = pd.read_csv(PRECAUTION_PATH)
precautions_map = precautions_df.set_index('Disease').T.to_dict('list')

# ========== Clean Text Function ==========
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========== Get Precautions ==========
def get_precautions(disease):
    return precautions_map.get(disease, ["N/A"])

# ========== Named Entity Recognition ==========
nlp = spacy.load("en_core_web_sm")
desc_df['clean_text'] = desc_df['Description'].apply(clean_text)
patterns = [nlp.make_doc(text) for text in desc_df['clean_text'].unique()]
matcher = PhraseMatcher(nlp.vocab)
matcher.add("SYMPTOM", patterns)

def extract_entities(text):
    doc = nlp(clean_text(text))
    matches = matcher(doc)
    return [doc[start:end].text for match_id, start, end in matches]

# ========== Streamlit UI ==========
st.set_page_config(page_title="Symptom-Based Disease Predictor", layout="centered")
st.title("🩺 Symptom-Based Disease Predictor")

user_input = st.text_area("Enter your symptoms (comma-separated or a sentence):", "fatigue, vomiting, yellow skin")

if st.button("Predict Disease"):
    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])
    prediction = model.predict(input_vec)[0]
    symptoms = extract_entities(user_input)
    precautions = get_precautions(prediction)

    st.success(f"🧠 Predicted Disease: **{prediction}**")

    st.markdown("### 🔍 Extracted Symptoms")
    if symptoms:
        for s in symptoms:
            st.markdown(f"- {s}")
    else:
        st.markdown("*No clear symptom matches found*")

    st.markdown("### 🛡️ Recommended Precautions")
    for p in precautions:
        st.markdown(f"- {p}")