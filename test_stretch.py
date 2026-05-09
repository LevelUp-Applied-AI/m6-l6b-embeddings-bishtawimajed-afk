import pytest
import os
import torch
import numpy as np
from stretch_embedding_explorer import get_distilbert_embedding

def test_data_files_exist():
    # Verify that the required data files are in the data directory
    assert os.path.exists("data/glove_50k_50d.txt"), "Glove file is missing!"
    assert os.path.exists("data/bbc_news.csv"), "BBC News file is missing!"

def test_distilbert_dimensions():
    # Verify the output dimension of DistilBERT (should be 768)
    sample_text = "Machine learning is fascinating."
    embedding = get_distilbert_embedding(sample_text)
    
    assert embedding.shape == (768,), f"Expected shape (768,), but got {embedding.shape}"

def test_embedding_not_empty():
    # Ensure the model is producing meaningful numerical data
    sample_text = "Testing some content"
    embedding = get_distilbert_embedding(sample_text)
    assert np.sum(np.abs(embedding)) > 0, "Embedding should not be all zeros!"