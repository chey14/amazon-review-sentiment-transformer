from data_preprocessing import load_and_prepare_data
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Load Data

print("Loading data...")
train_texts, test_texts, train_labels, test_labels = load_and_prepare_data()

# Reduced Size
train_texts = train_texts[:20000]
train_labels = train_labels[:20000]

test_texts = test_texts[:5000]
test_labels = test_labels[:5000]

print("Data ready.")

# Tokenization

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

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

train_dataset = ReviewDataset(train_encodings, train_labels)
test_dataset = ReviewDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Device Setup

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Load Model

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5
)

model.to(device)

# Optimizer

optimizer = AdamW(model.parameters(), lr=5e-5)

# Traning Loop

epochs = 2

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()

    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print("Average training loss:", avg_loss)

print("Training complete.")

# Model Save

model.save_pretrained("models/sentiment_model")
tokenizer.save_pretrained("models/sentiment_model")

print("Model saved.")