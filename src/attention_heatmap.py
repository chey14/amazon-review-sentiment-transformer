import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "models/sentiment_model"

model = BertForSequenceClassification.from_pretrained(
    model_path,
    output_attentions=True
)

tokenizer = BertTokenizer.from_pretrained(model_path)

sentence = "This product is very bad and stopped working"

inputs = tokenizer(sentence, return_tensors="pt")

outputs = model(**inputs)

attentions = outputs.attentions

# take last layer attention
attention = attentions[-1][0][0].detach().numpy()

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

plt.figure(figsize=(10,8))
sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="Blues")
plt.title("BERT Attention Map")
plt.show()