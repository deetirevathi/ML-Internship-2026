# ==============================
# CUSTOMER SUPPORT TICKETS ML PIPELINE
# Steps 4–7: Text Cleaning, Vectorization, Model Training, Evaluation
# ==============================

import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# STEP 1: Load Dataset
# ------------------------------
df = pd.read_csv("customer_support_tickets.csv")
print("Dataset Loaded. Shape:", df.shape)
print("Columns:", df.columns)  # check columns

# ------------------------------
# STEP 2: Stopwords
# ------------------------------
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
])

# ------------------------------
# STEP 3 / STEP 4: Clean & Preprocess Text
# ------------------------------
def clean_text(text):
    text = str(text).lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()  # simple tokenizer
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return " ".join(tokens)

# Use the correct text column from your CSV
text_column = 'Ticket Description'
df['cleaned_text'] = df[text_column].apply(clean_text)
print("Text cleaning completed.")

# ------------------------------
# STEP 5: Convert Text to Numbers (TF-IDF)
# ------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'])
print("TF-IDF Vectorization done. Shape of X:", X.shape)

# ------------------------------
# STEP 6: Train ML Models
# ------------------------------
# Targets
y_category = df['Ticket Type']      # adjust if your CSV has a different column
y_priority = df['Ticket Priority']

# Split data into train/test
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_category, test_size=0.2, random_state=42)
X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(X, y_priority, test_size=0.2, random_state=42)

# Train Multinomial Naive Bayes models
model_category = MultinomialNB()
model_category.fit(X_train_cat, y_train_cat)

model_priority = MultinomialNB()
model_priority.fit(X_train_pri, y_train_pri)

print("Models trained successfully.")

# ------------------------------
# STEP 7: Evaluate Performance
# ------------------------------
# Category Model Evaluation
y_pred_cat = model_category.predict(X_test_cat)
print("===== Category Model Performance =====")
print("Accuracy:", accuracy_score(y_test_cat, y_pred_cat))
print(classification_report(y_test_cat, y_pred_cat))

# Priority Model Evaluation
y_pred_pri = model_priority.predict(X_test_pri)
print("===== Priority Model Performance =====")
print("Accuracy:", accuracy_score(y_test_pri, y_pred_pri))
print(classification_report(y_test_pri, y_pred_pri))