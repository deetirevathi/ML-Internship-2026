# ==============================
# RESUME MATCHING PIPELINE (WORKING)
# ==============================

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# STEP 1: Load Resume Dataset
# ------------------------------
df_resume = pd.read_csv("Resume/Resume.csv")  # adjust path if needed
print("Dataset loaded. Shape:", df_resume.shape)
print("Columns:", df_resume.columns)

# Correct column name for resume text
resume_column = 'Resume_str'

# ------------------------------
# STEP 2: NLTK Setup
# ------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------------------
# STEP 3: Clean Resume Text
# ------------------------------
def clean_text(text):
    text = str(text).lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = ''.join([i for i in text if not i.isdigit()])  # remove numbers
    tokens = text.split()  # simple tokenizer
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df_resume['clean_text'] = df_resume[resume_column].apply(clean_text)
print("Resume text cleaning completed.")

# ------------------------------
# STEP 4: Extract Skills (Custom List)
# ------------------------------
skill_keywords = [
    'python', 'sql', 'excel', 'machine learning', 'data visualization',
    'statistics', 'r', 'tableau', 'power bi', 'java', 'c++', 'aws',
    'deep learning', 'nlp', 'pandas', 'numpy', 'scikit-learn', 'keras'
]

def extract_skills(text):
    text = text.lower()
    skills_found = [skill for skill in skill_keywords if skill in text]
    return skills_found

df_resume['skills'] = df_resume['clean_text'].apply(extract_skills)

# ------------------------------
# STEP 5: Define Job Description
# ------------------------------
job_description = """
Looking for a Data Analyst skilled in Python, SQL, Excel,
Machine Learning, Data Visualization, and Statistics.
"""

def clean_job_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

clean_job_description = clean_job_text(job_description)
jd_skills = extract_skills(clean_job_description)

# ------------------------------
# STEP 6: TF-IDF Vectorization
# ------------------------------
tfidf = TfidfVectorizer()
resume_vectors = tfidf.fit_transform(df_resume['clean_text'])
jd_vector = tfidf.transform([clean_job_description])

# ------------------------------
# STEP 7: Compute Similarity Score
# ------------------------------
scores = cosine_similarity(resume_vectors, jd_vector).flatten()
df_resume['match_score'] = scores

# ------------------------------
# STEP 8: Rank Candidates
# ------------------------------
df_ranked = df_resume.sort_values(by='match_score', ascending=False)

# ------------------------------
# STEP 9: Identify Missing Skills
# ------------------------------
df_ranked['missing_skills'] = df_ranked['skills'].apply(lambda x: list(set(jd_skills) - set(x)))

# ------------------------------
# Display Top Candidates
# ------------------------------
top_candidates = df_ranked[['Resume_str', 'skills', 'missing_skills', 'match_score']].head(10)
print(top_candidates)