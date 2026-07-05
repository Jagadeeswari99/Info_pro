"""
FastAPI Application - AI Citizen Grievance Analysis System
Week 4 Deployment - Infotact Internship Project 1

As per project guidelines:
- Accepts JSON payload with citizen's raw text complaint
- Returns predicted department and priority score
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk, re

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Driven Citizen Grievance & Sentiment Analysis System",
    description="Automatically routes citizen complaints to departments and assigns urgency priority scores.",
    version="1.0.0"
)

# Load or train model at startup
print("Loading model...")
try:
    pipeline = joblib.load("complaint_model.pkl")
    print("Model loaded from file!")
except:
    print("Training model from scratch...")
    df = pd.read_csv("cleaned_data.csv")
    top10 = df['Complaint Type'].value_counts().head(10).index
    df = df[df['Complaint Type'].isin(top10)]
    df['combined_text'] = df['clean_text'] + ' ' + df['Complaint Type'].str.lower()
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
        ('model', LinearSVC(class_weight='balanced', max_iter=2000, random_state=42))
    ])
    pipeline.fit(df['combined_text'], df['Complaint Type'])
    print("Model trained!")

# Preprocessing setup
stop_words = set(stopwords.words('english')) - {'no', 'not', 'never', 'neither', 'nor'}
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

KEYWORD_RULES = {
    'Derelict Vehicle':        ['abandoned car', 'derelict', 'rusting', 'no license plate', 'junk car'],
    'Noise - Street/Sidewalk': ['loud music', 'noise', 'shouting', 'fighting loudly', 'loud party', 'screaming', 'yelling'],
    'Blocked Driveway':        ['blocking my driveway', 'blocked driveway', 'cant get out', 'blocking my gate'],
    'Graffiti':                ['graffiti', 'spray paint', 'vandal'],
    'Animal Abuse':            ['animal abuse', 'beating dog', 'kicking cat', 'hurting animal', 'kicking a dog', 'stray dog'],
    'Illegal Parking':         ['illegally parked', 'illegal parking', 'fire hydrant', 'double parked'],
    'Traffic':                 ['traffic light', 'traffic signal', 'street light', 'pothole', 'road broken', 'no lights'],

    # ✅ ADD THESE TWO NEW ENTRIES:
    'Disorderly Youth':        ['bullying', 'teenagers', 'gang of boys', 'disorderly youth',
                                'racing motorbike', 'vandali', 'blocking entrance', 'threatening residents'],
    'Homeless Encampment':     ['homeless', 'encampment', 'sleeping on footpath', 'temporary shelter'],
}

def keyword_override(text: str):
    text_lower = text.lower()
    for dept, keywords in KEYWORD_RULES.items():
        if any(kw in text_lower for kw in keywords):
            return dept
    return None  # no match → use model


NEGATIVE_KEYWORDS = [
    'fighting', 'shouting', 'yelling', 'screaming', 'threatening',
    'abusing', 'drunk', 'harassing', 'attacking', 'beating',
    'violent', 'dangerous', 'illegal', 'broken', 'damaged',
    'blocked', 'abandoned', 'injured', 'accident', 'abuse',
    'cannot', 'unable', 'no one', 'nobody', 'ignored',
    'bullying', 'bully', 'vandali', 'racing',   # ✅ ADD THIS LINE
]

# Request Model - JSON input
class ComplaintRequest(BaseModel):
    complaint: str

    class Config:
        json_schema_extra = {
            "example": {
                "complaint": "There is very loud music coming from the apartment next door since midnight."
            }
        }


# Response Model - JSON output
class ComplaintResponse(BaseModel):
    complaint: str
    department: str
    sentiment: str
    polarity: float
    priority_score: float
    priority_label: str


# Health check endpoint
@app.get("/")
def home():
    return {
        "status": "running",
        "project": "AI-Driven Citizen Grievance & Sentiment Analysis System",
        "usage": "POST /predict with JSON body: {complaint: 'your complaint text'}",
        "docs": "Visit /docs for interactive API documentation"
    }


# Main prediction endpoint
# As per guidelines: accepts JSON payload, returns department and priority score
@app.post("/predict", response_model=ComplaintResponse)
def predict(request: ComplaintRequest):

    # Step 1: Preprocess the raw complaint
    cleaned = preprocess(request.complaint)

    # Step 2: Predict department — keyword override first, model as fallback
    # FIX 3: moved cleaned above, and keyword_override result is NOT overwritten
    department = keyword_override(request.complaint) or pipeline.predict([cleaned])[0]

    # Step 3: Sentiment analysis on original text
    blob = TextBlob(request.complaint)
    polarity = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    # Step 4: Override polarity if complaint has strong negative context words
    # (TextBlob sometimes gives positive score to negative sentences)
    negative_hits = [w for w in NEGATIVE_KEYWORDS if w in request.complaint.lower()]
    if negative_hits and polarity > 0:
        polarity = -0.2  # force to Negative

    # Step 5: Classify sentiment
    if polarity < -0.5:
        sentiment = "Critical/Urgent"
    elif polarity < 0:
        sentiment = "Negative"
    elif polarity == 0:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    # Step 6: Calculate priority score
    priority_score = 5 + (-polarity * 3) + (subjectivity * 2)
    priority_score = round(min(max(priority_score, 1), 10), 2)

    # Step 7: Assign priority label
    if priority_score >= 7:
        priority_label = "HIGH PRIORITY - Immediate Action Required"
    elif priority_score >= 4:
        priority_label = "MEDIUM PRIORITY - Action within 24-48 hours"
    else:
        priority_label = "LOW PRIORITY - Routine Response"

    return ComplaintResponse(
        complaint=request.complaint,
        department=department,
        sentiment=sentiment,
        polarity=polarity,
        priority_score=priority_score,
        priority_label=priority_label
    )


# Run with: uvicorn main:app --reload
# Test at: http://127.0.0.1:8000/docs
