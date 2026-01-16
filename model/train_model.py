import pandas as pd
import nltk
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- NLTK SETUP ----------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ---------------- LOAD DATA ----------------
df = pd.read_csv("../dataset/phishing_emails.csv")

print("Original shape:", df.shape)
print("Columns:", df.columns)

# --------- COLUMN NORMALIZATION ----------
possible_text_cols = ['text', 'email', 'Email Text', 'body', 'content', 'text_combined']
possible_label_cols = ['label', 'Label', 'class']

text_col = next((c for c in possible_text_cols if c in df.columns), None)
label_col = next((c for c in possible_label_cols if c in df.columns), None)

if text_col is None or label_col is None:
    raise ValueError("Text or label column not found in dataset")

df = df.rename(columns={text_col: 'text', label_col: 'label'})

# ---------------- MISSING VALUE HANDLING ----------------
# Handle missing email text → replace with empty string
df['text'] = df['text'].fillna("")

# Handle invalid labels → keep only 0 and 1
df = df[df['label'].isin([0, 1])]
df['label'] = df['label'].astype(int)

print("Shape after cleaning:", df.shape)
print(df['label'].value_counts())

# ---------------- NLP PREPROCESSING ----------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned_text'] = df['text'].apply(clean_text)

# ---------------- FEATURE EXTRACTION ----------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# ---------------- TRAIN / VAL / TEST SPLIT ----------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ---------------- MODEL TRAINING ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully")
