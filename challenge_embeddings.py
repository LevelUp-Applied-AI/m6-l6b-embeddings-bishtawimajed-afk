import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from transformers import AutoTokenizer

def run_tier1_normalization(tfidf_matrix, glove_embs, bert_embs):
    tfidf_norm = normalize(tfidf_matrix, norm='l2')
    glove_norm = normalize(glove_embs, norm='l2')
    bert_norm = normalize(bert_embs, norm='l2')
    
    query_idx = 0
    cos_sim = cosine_similarity(bert_norm[query_idx:query_idx+1], bert_norm).flatten()
    euc_dist = euclidean_distances(bert_norm[query_idx:query_idx+1], bert_norm).flatten()
    
    cos_rank = np.argsort(cos_sim)[::-1][:10]
    euc_rank = np.argsort(euc_dist)[:10]
    
    match = np.array_equal(cos_rank, euc_rank)
    print(f"Tier 1 - L2 Normalization Success: {match}")
    return match

def run_tier2_tokenization(texts, model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    subword_counts = []
    word_fragment_map = {}

    for text in texts[:200]:
        words = text.split()
        for word in words:
            tokens = tokenizer.tokenize(word)
            subword_counts.append(len(tokens))
            if len(tokens) > 1:
                word_fragment_map[word] = word_fragment_map.get(word, 0) + 1
                
    avg_subwords = np.mean(subword_counts)
    top_fragmented = sorted(word_fragment_map.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"Tier 2 - Avg subwords: {avg_subwords:.2f}")
    print(f"Tier 2 - Top fragmented: {top_fragmented}")
    return avg_subwords, top_fragmented

def run_tier3_evaluation(relevant_indices, ranked_indices):
    mrr = 0
    for rank, idx in enumerate(ranked_indices, 1):
        if idx in relevant_indices:
            mrr = 1 / rank
            break
            
    p_at_5 = len(set(ranked_indices[:5]) & set(relevant_indices)) / 5
    print(f"Tier 3 - MRR: {mrr:.3f} | P@5: {p_at_5:.2f}")
    return mrr, p_at_5

if __name__ == "__main__":
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    
    # Tier 1 Execution (using dummy matrices for structure)
    t_fake, g_fake, b_fake = np.random.rand(100, 100), np.random.rand(100, 50), np.random.rand(100, 768)
    run_tier1_normalization(t_fake, g_fake, b_fake)
    
    # Tier 2 Execution
    run_tier2_tokenization(texts)
    
    # Tier 3 Execution
    ground_truth = [1, 5, 10]
    predictions = [2, 1, 8, 5, 12, 10]
    run_tier3_evaluation(ground_truth, predictions)