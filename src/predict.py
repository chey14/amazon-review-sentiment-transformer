import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load saved model
model_path = "models/sentiment_model"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model.to(device)
model.eval()

# Label mapping
labels_map = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

def predict_review(review_text):
    encoding = tokenizer(
        review_text,
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


# Testing custom Reviews

if __name__ == "__main__":
    while True:
        review = input("\nEnter a review (or type 'exit'): ")

        if review.lower() == "exit":
            break

        sentiment, confidence = predict_review(review)

        print(f"\nPredicted Sentiment: {sentiment}")
        print("Confidence Scores:")
        print(f"Very Negative: {confidence[0]:.4f}")
        print(f"Negative     : {confidence[1]:.4f}")
        print(f"Neutral      : {confidence[2]:.4f}")
        print(f"Positive     : {confidence[3]:.4f}")
        print(f"Very Positive: {confidence[4]:.4f}")