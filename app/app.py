import streamlit as st
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    layout="centered"
)

st.title("🛍 Amazon Review Sentiment Analysis")
st.markdown("### Transformer-based Sentiment Classification using BERT")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@st.cache_resource
def load_model():
    model_path = "models/sentiment_model"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

labels_map = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}


def predict_review(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    return labels_map[predicted_class], probs.cpu().numpy()[0]




st.header("🔍 Analyze Single Review")

user_input = st.text_area("Enter a product review:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        sentiment, confidence = predict_review(user_input)

        
        if sentiment == "Positive":
            st.success(f"Predicted Sentiment: {sentiment}")
        elif sentiment == "Negative":
            st.error(f"Predicted Sentiment: {sentiment}")
        else:
            st.warning(f"Predicted Sentiment: {sentiment}")

        
        st.subheader("📊 Confidence Scores")

        confidence_df = pd.DataFrame({
            "Sentiment": [
            "Very Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Very Positive"
            ],
            "Confidence": confidence
        })

        st.bar_chart(confidence_df.set_index("Sentiment"))

    else:
        st.warning("Please enter a review.")




st.header("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload a CSV file with a column named 'review_text'",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review_text" in df.columns:
        st.info("Processing reviews...")

        df["Predicted_Sentiment"] = df["review_text"].apply(
            lambda x: predict_review(str(x))[0]
        )

        st.subheader("🔎 Preview")
        st.dataframe(df.head())

        
        sentiment_counts = df["Predicted_Sentiment"].value_counts()

        st.subheader("📊 Sentiment Distribution (Bar Chart)")

        chart_df = sentiment_counts.reset_index()
        chart_df.columns = ["Sentiment", "Count"]

        st.bar_chart(chart_df.set_index("Sentiment"))

        
        st.subheader("🥧 Sentiment Share (Pie Chart)")

        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%"
        )
        ax.axis("equal")  

        st.pyplot(fig)

    else:
        st.error("CSV must contain a column named 'review_text'")