import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader
from data_preprocessing import load_and_prepare_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Device Setup

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model

model_path = "models/sentiment_model"

print("Loading model...")
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()
print("Model loaded successfully.")

# Load Test Data

_, test_texts, _, test_labels = load_and_prepare_data()
test_texts = test_texts[:5000]
test_labels = test_labels[:5000]

print("Tokenizing test data...")

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=128
)

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = ReviewDataset(test_encodings, test_labels)


test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Number of batches:", len(test_loader))
print("Number of samples:", len(test_dataset))

print("Starting evaluation...")

all_preds = []
all_labels = []


# Evaluation Loop

with torch.no_grad():
    for batch in tqdm(test_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Metrics

accuracy = accuracy_score(all_labels, all_preds)

print("\n==========================")
print(f"Test Accuracy: {accuracy:.4f}")
print("==========================\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds))


import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Convert labels to binary format
y_true = label_binarize(all_labels, classes=[0,1,2,3,4])
y_pred = label_binarize(all_preds, classes=[0,1,2,3,4])

n_classes = y_true.shape[1]

plt.figure()

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()




# Confusion Matrix

cm = confusion_matrix(all_labels, all_preds)

labels_map = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive"
]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_map,
    yticklabels=labels_map
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()