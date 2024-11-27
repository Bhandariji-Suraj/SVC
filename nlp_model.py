import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import re

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_query(query):
    """Preprocess the user query by removing punctuation and converting to lowercase."""
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    return query

def get_query_embedding(query):
    """Generate embeddings for the user query."""
    query = preprocess_query(query)
    return model.encode(query)

def find_top_matches(query_embedding, dataset, top_n=3, threshold=0.5):
    """
    Find top matching results based on cosine similarity.

    Args:
        query_embedding: Embedding of the user's query.
        dataset: DataFrame containing embeddings and metadata.
        top_n: Number of top matches to return.
        threshold: Minimum similarity score to consider a match.

    Returns:
        DataFrame of top matching results.
    """
    similarities = [
        1 - cosine(query_embedding, np.array(embedding)) for embedding in dataset['embeddings']
    ]
    dataset['similarity'] = similarities
    top_results = dataset[dataset['similarity'] >= threshold]
    return top_results.sort_values(by='similarity', ascending=False).head(top_n)
