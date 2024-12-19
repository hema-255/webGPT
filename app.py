
import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the FAISS index, metadata, and chunked data
with open("faiss_index.pkl", "rb") as f:
    index, metadata, chunked_data = pickle.load(f)

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to retrieve relevant chunks
def retrieve_chunks(query, top_k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    results = [(metadata[i][0], chunked_data[metadata[i][0]][metadata[i][1]]) for i in indices[0]]
    return results

# Streamlit UI
st.title("WebGPT")
st.markdown("Ask me anything about the ingested content!")

# Input from user
user_query = st.text_input("Enter your question:", "")

def beautify_answer(response):
    justified_text = f'<div style="text-align: justify; font-size: 16px; line-height: 1.6;">{response}</div>'
    return justified_text


if user_query:
    st.markdown("### Retrieved Context:")
    retrieved_chunks = retrieve_chunks(user_query)
    for i, (url, chunk) in enumerate(retrieved_chunks):
        st.write(f"**Source {i+1}:** {url}")
        st.markdown(beautify_answer(chunk), unsafe_allow_html=True)
        st.write("\n")

    # Generate response (basic concatenation for now)
    response = " ".join([str(chunk) for url, chunk in retrieved_chunks])  # Ensure only chunks are concatenated
    # Fallback if no valid chunks
    if not response.strip():
      response = "Sorry, I couldn't find relevant information to answer your query."
    st.markdown("### Answer:")
    st.markdown(beautify_answer(response), unsafe_allow_html=True)
