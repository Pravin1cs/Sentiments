import nltk
nltk.download('vader_lexicon')
from supabase import create_client, Client
from transformers import pipeline
import pandas as pd
import string
from fastapi import FastAPI
from nltk.sentiment.vader import SentimentIntensityAnalyzer

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# Replace with your actual values
SUPABASE_URL = "https://bvtirlkozgihvapfzovn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ2dGlybGtvemdpaHZhcGZ6b3ZuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDE5NDY5NCwiZXhwIjoyMDU5NzcwNjk0fQ.i2NCQCBiE6vzQ8rZbAHSxewjrfuOsYCE28J2xdtnwA8"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


response = supabase.table("sentiment_analysis").select("*").execute()

#print("Data:", response.data)
df = pd.DataFrame(response.data)
print(df)
df2 = pd.DataFrame({'transcript': [], 'Sentiment': []})


for index, row in df.iterrows():
    text =row['transcript']
    lower_case = text.lower()
    lower_case = lower_case.replace('-', ' ')
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    summary = summarizer(text, max_length=200, min_length=25, do_sample=False)[0]['summary_text']
    score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
    if score['neg'] > score['pos']:
        stt = "Negative Sentiment"
        scr =score['pos']-score['neg']
    elif score['neg'] < score['pos']:
        stt = "Positive Sentiment"
        scr = score['pos'] - score['neg']
    else:
        stt = "Neutral Sentiment"
        scr = score['pos'] - score['neg']
    update_response = supabase.table("customer_interactions").update({
        "sentiment_score": scr, "sentiment": stt, "sentiment_analysis": summary
    }).eq("id", row["id"]).execute()

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Sentiment API is up and running!"}
