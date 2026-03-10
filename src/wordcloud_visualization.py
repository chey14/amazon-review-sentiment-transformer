import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from data_preprocessing import load_and_prepare_data

print("Loading data...")

train_texts, _, _, _ = load_and_prepare_data()


train_texts = train_texts[:5000]   # 5000 reviews

print("Generating word cloud...")

text = " ".join(train_texts)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud of Amazon Reviews")
plt.show()