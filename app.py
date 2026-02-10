import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

# Streamlit UI
st.set_page_config(page_title="Amazon Review Sentiment Analysis", layout="centered")

st.title("🛒 Amazon Review Sentiment Analysis")
st.write("Enter a product review to predict sentiment.")

review = st.text_area("✍️ Enter Review Text", height=150)

if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_review = clean_text(review)
        vectorized_review = vectorizer.transform([clean_review])
        prediction = model.predict(vectorized_review)[0]

        if prediction == "Positive":
            st.success("😊 Sentiment: Positive")
        elif prediction == "Negative":
            st.error("😠 Sentiment: Negative")
        else:
            st.info("😐 Sentiment: Neutral")
