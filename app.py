import streamlit as st
from transformers import pipeline

# ✅ IMPORTANT: use FULL PATH (replace "user" if needed)
MODEL_PATH = r"C:\Users\user\Desktop\sentiment_model"

# Load local model + tokenizer
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

# UI
st.title("🧠 Sentiment Analysis App")
st.write("Enter text and get prediction (Positive / Negative)")

# Input box
text = st.text_area("Enter your text:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        result = sentiment_pipeline(text)[0]

        label = result["label"]
        score = result["score"]

        # Emoji logic
        emoji = "😊" if label == "POSITIVE" else "😡"

        st.success(f"Result: {label} {emoji}")
        st.info(f"Confidence: {score:.4f}")