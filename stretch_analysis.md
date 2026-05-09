# Embedding Space Explorer Analysis

## 1. Dimensionality Reduction Choice
For this analysis, I applied **t-SNE** for word embeddings and **PCA** for document embeddings. 
- **t-SNE** was chosen for GloVe vectors because it excels at preserving local structures, which effectively revealed tight semantic clusters like "Sports" and "Emotions."
- **PCA** was used for the BBC News DistilBERT embeddings to capture the global variance between broad news categories, making it easier to see how major topics like "Politics" and "Sport" diverge across the entire space.

## 2. Word Embedding Observations (GloVe)
The t-SNE plot reveals clear semantic grouping:
- **Clusters:** There are distinct islands for each category. For example, the "Technology" cluster (green) shows words like *software* and *artificial* in close proximity, while the "Finance" cluster (yellow) groups *bank* and *currency* together.
- **Outliers:** Some words like *marathon* appear on the edge of the "Sports" cluster, likely because they bridge the gap between specific sports events and general activities.
- **Relationship:** The distance between the "Emotions" and "Finance" clusters confirms that the model correctly separates abstract human feelings from concrete economic concepts.

## 3. Document Embedding Observations (DistilBERT)
The PCA visualization of BBC News articles demonstrates that high-dimensional model outputs retain strong categorical intent:
- **Separation:** "Sport" articles (orange) are strongly clustered at the bottom right, showing high internal similarity. "Politics" (blue) is located on the opposite left side.
- **Overlaps:** We can see some proximity between "Business" (red) and "Tech" (green), which is expected as many technology news articles often discuss market trends, trade gaps, or corporate acquisitions.
- **Semantic Logic:** Articles about *unit unveils pol...* (Politics) are far from articles like *sony psp console hit...* (Tech), validating that DistilBERT successfully captures the context of the entire article.

## 4. Conclusion
Both visualizations confirm that whether at the word level or the document level, embeddings are not random. They capture a structured "map" of human knowledge where semantic similarity translates directly into spatial proximity.