import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer, AutoModel

# --- PART 1: GLOVE WORD EMBEDDINGS ---

categories = {
    "Sports": ["football", "basketball", "tennis", "soccer", "swimming", "athlete", "olympics", "tournament", "stadium", "referee", 
               "marathon", "volleyball", "gymnastics", "wrestling", "baseball", "cricket", "rugby", "boxing", "medal", "stadium"],
    "Countries": ["jordan", "france", "japan", "brazil", "canada", "egypt", "germany", "italy", "china", "australia",
                 "mexico", "spain", "india", "russia", "argentina", "turkey", "greece", "thailand", "korea", "sweden"],
    "Technology": ["software", "hardware", "internet", "robot", "computer", "algorithm", "database", "network", "server", "coding",
                  "artificial", "intelligence", "mobile", "encryption", "firewall", "browser", "python", "javascript", "developer", "automation"],
    "Emotions": ["happy", "sad", "angry", "excited", "lonely", "joy", "fear", "brave", "confused", "bored",
                "guilt", "shame", "surprised", "proud", "jealous", "disappointed", "hopeful", "calm", "anxiety", "cheerful"],
    "Finance": ["bank", "money", "economy", "stock", "investment", "market", "profit", "budget", "tax", "trading",
               "currency", "inflation", "revenue", "bankruptcy", "dividend", "wallet", "insurance", "pension", "finance", "debt"]
}

def load_glove_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

glove_path = "data/glove_50k_50d.txt"
word_vectors = load_glove_embeddings(glove_path)

all_words = []
all_vectors = []
word_labels = []

for category, words in categories.items():
    for word in words:
        if word in word_vectors:
            all_words.append(word)
            all_vectors.append(word_vectors[word])
            word_labels.append(category)

X_words = np.array(all_vectors)

tsne_words = TSNE(n_components=2, perplexity=15, random_state=42, init='pca', learning_rate='auto')
X_words_2d = tsne_words.fit_transform(X_words)

plt.figure(figsize=(12, 8))
colors = {'Sports': 'red', 'Countries': 'blue', 'Technology': 'green', 'Emotions': 'purple', 'Finance': 'orange'}

for category in categories.keys():
    indices = [i for i, label in enumerate(word_labels) if label == category]
    plt.scatter(X_words_2d[indices, 0], X_words_2d[indices, 1], c=colors[category], label=category, alpha=0.7)

for i in range(0, len(all_words), len(all_words)//10):
    plt.annotate(all_words[i], (X_words_2d[i, 0], X_words_2d[i, 1]), xytext=(5, 2), textcoords='offset points', fontsize=9)

plt.legend()
plt.title("GloVe Word Embeddings (t-SNE)")
plt.savefig("word_embeddings_plot.png")
plt.show()

# --- PART 2: BBC NEWS DISTILBERT EMBEDDINGS ---

df = pd.read_csv("data/bbc_news.csv")
# Select 4 articles per category to get exactly 20
sample_df = df.groupby('category').head(4).reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_distilbert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

print("Computing DistilBERT embeddings for BBC articles...")
sample_df['embeddings'] = sample_df['text'].apply(get_distilbert_embedding)

X_docs = np.stack(sample_df['embeddings'].values)

# PCA is often better for preserving global structure in document sets
pca_docs = PCA(n_components=2)
X_docs_2d = pca_docs.fit_transform(X_docs)

plt.figure(figsize=(12, 8))
doc_categories = sample_df['category'].unique()

for cat in doc_categories:
    idx = sample_df[sample_df['category'] == cat].index
    plt.scatter(X_docs_2d[idx, 0], X_docs_2d[idx, 1], label=cat, s=100)
    
    # Updated annotation logic to use first 20 chars of 'text' instead of 'title'
    for i in idx:
        label_text = sample_df['text'].iloc[i][:20] + "..."
        plt.annotate(label_text, (X_docs_2d[i, 0], X_docs_2d[i, 1]), fontsize=8, alpha=0.7)

plt.legend()
plt.title("BBC News Document Embeddings (PCA)")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.savefig("document_embeddings_plot.png")
plt.show()