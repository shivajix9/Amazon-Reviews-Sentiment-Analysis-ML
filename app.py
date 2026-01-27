import streamlit as st
import joblib
import re
import string

# -------------------------
# Load model & vectorizer
# -------------------------
model = joblib.load("best_sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# -------------------------
# Text cleaning function
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Amazon Review Sentiment", page_icon="⭐", layout="centered")

st.title("⭐ Amazon Review Sentiment Analyzer")
st.write("Enter a product review below to predict its sentiment.")

review_text = st.text_area(
    "✍️ Enter your review:",
    height=150,
    placeholder="This product is amazing! The quality exceeded my expectations..."
)

if st.button("🔍 Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review text.")
    else:
        cleaned = clean_text(review_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        sentiment_map = {
            0: "Negative 😠",
            1: "Neutral 😐",
            2: "Positive 😊"
        }

        st.subheader("📊 Prediction Result")
        st.success(f"**Sentiment:** {sentiment_map.get(prediction, prediction)}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
