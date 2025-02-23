import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO

st.title('Reuters Article Similarity Finder')

# Load article embeddings and text data
document_embeddings = np.load("document_embeddings.npy")
document_embeddings = document_embeddings[:500]
reuters_articles = np.load("articles.npy", allow_pickle=True)

st.write(f"Successfully loaded {len(reuters_articles)} articles.")

# Dropdown to select a model
available_models = ['Model A', 'Model B', 'Model C']
chosen_model = st.selectbox('Choose a model:', available_models)

# User input for text
input_text = st.text_input('Enter text to compare:')

# Button to trigger similarity computation
if st.button('Find Similar Articles'):
    if input_text.strip():
        # Generate a random user input embedding for demonstration
        input_vector = np.random.rand(1, document_embeddings.shape[1])

        # Calculate cosine similarity between user input and document embeddings
        similarity_scores = cosine_similarity(input_vector, document_embeddings)

        # Retrieve top-k most similar articles
        top_results = 5
        most_similar_indexes = np.argsort(similarity_scores[0])[-top_results:][::-1]

        # Show snippets of top-k similar articles
        st.write("Top 5 most relevant article snippets:")
        for idx in most_similar_indexes:
            article_snippet = reuters_articles[idx][:50]  # First 50 characters of the article
            st.write(f" `{article_snippet}...`")
    else:
        st.warning("Please provide some text to compare.")
