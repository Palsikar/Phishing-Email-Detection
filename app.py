import streamlit as st
import pandas as pd
import joblib
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix

# ---------------- NLTK SETUP ----------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/phishing_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset/phishing_emails.csv")

# --------- COLUMN NORMALIZATION ----------
possible_text_cols = ['text', 'email', 'Email Text', 'body', 'content', 'text_combined']
possible_label_cols = ['label', 'Label', 'class']

text_col = next((c for c in possible_text_cols if c in df.columns), None)
label_col = next((c for c in possible_label_cols if c in df.columns), None)

if text_col is None or label_col is None:
    st.error("Dataset columns not detected correctly.")
    st.stop()

df = df.rename(columns={text_col: 'text', label_col: 'label'})

# ---------------- MISSING VALUE HANDLING ----------------
df['text'] = df['text'].fillna("")
df = df[df['label'].isin([0, 1])]
df['label'] = df['label'].astype(int)

# ---------------- NLP TOOLS ----------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Phishing Email Detection", layout="centered")

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", ["🔍 Email Detection", "📈 Model Evaluation"])

# -------- PAGE 1: DETECTION --------
if page == "🔍 Email Detection":
    st.title("📧 Phishing Email Detection System")

    email_text = st.text_area("Paste Email Content", height=220)

    if st.button("Detect Email"):
        if email_text.strip() == "":
            st.warning("⚠️ Please enter email content.")
        else:
            cleaned = clean_text(email_text)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0][1]

            if prediction == 1:
                st.error(f"🚨 Phishing Email Detected\n\nProbability: {probability:.2f}")
            else:
                st.success(f"✅ Legitimate Email\n\nConfidence: {1 - probability:.2f}")

# -------- PAGE 2: EVALUATION --------
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")

    df['cleaned_text'] = df['text'].apply(clean_text)
    X = vectorizer.transform(df['cleaned_text'])
    y_true = df['label']
    y_pred = model.predict(X)

    accuracy = (y_true == y_pred).mean()
    st.metric("Model Accuracy", f"{accuracy:.2%}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

st.markdown("---")
st.caption("Major Project | Phishing Email Detection using NLP & Machine Learning")
