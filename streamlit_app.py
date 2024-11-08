import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import streamlit as st
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from pathlib import Path

# Streamlit configuration for a light theme and custom title
st.set_page_config(page_title="Text Embedding Manager", layout="centered")

# CSS for a more polished look
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.title("Text Embedding Manager")
st.subheader("A streamlined tool for creating, searching, and managing text embeddings")
st.write("Use this app to upload text files, generate sentence embeddings, and find similar sentences. Enjoy a light theme and a refined layout for ease of use.")

# Class containing functions needed to create sentence embeddings
class TextEmbeddingManager:
    def __init__(self, directory_path=None):
        self.directory_path = directory_path
        self.df = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    # Comb through directory and grab .txt files
    def collect_txt_files_data(self, uploaded_files):
        filepaths, contents = [], []
        for file in uploaded_files:
            filepaths.append(file.name)
            contents.append(file.getvalue().decode("utf-8"))

        self.df = pd.DataFrame({
            'filepath': filepaths,
            'transcript': contents
        })

    # Use MiniLM to encode all sentences in .txt files
    def encode_sentences_sbert(self):
        if self.df is None:
            raise ValueError("No data found. Please specify a directory with text files first.")

        self.df['embeddings'] = None
        progress_bar = st.progress(0)  # Initialize the progress bar
        for index, row in self.df.iterrows():
            text = str(row['transcript'])
            sentences = sent_tokenize(text)  # Use NLTK's sent_tokenize for comprehensive sentence splitting
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            
            # Ensure sentences list is not empty before encoding
            if sentences:
                doc_embeddings = self.model.encode(sentences)
                self.df.at[index, 'embeddings'] = [embedding.tolist() for embedding in doc_embeddings]
            else:
                self.df.at[index, 'embeddings'] = []  # Store empty list if no valid sentences

            # Update progress bar
            progress_bar.progress((index + 1) / len(self.df))
        progress_bar.empty()  # Clear the progress bar

    # Save embeddings as a pickle file
    def save_data_pickle(self):
        return pickle.dumps(self.df)

    # Load filepaths, text, and embedding lists from pickle
    def load_data_pickle(self, pickle_data):
        self.df = pickle.loads(pickle_data)

    # Calculate cosine similarity to find closest semantic match to target sentence
    def find_and_save_closest_sentences(self, target_sentence):
        if self.df is None or 'embeddings' not in self.df.columns:
            raise ValueError("Embeddings not generated. Please generate embeddings first.")

        target_embedding = self.model.encode([target_sentence])[0]

        results = []
        for index, row in self.df.iterrows():
            sentence_embeddings = np.array(row['embeddings'])
            sentences = sent_tokenize(row['transcript'])  # Use NLTK's sent_tokenize for comprehensive sentence splitting
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

            if not sentences:
                continue

            # Calculate similarity score
            similarities = [np.dot(target_embedding, embedding) /
                            (np.linalg.norm(target_embedding) * np.linalg.norm(embedding)) for embedding in sentence_embeddings]
            max_similarity_index = np.argmax(similarities)
            max_similarity_score = similarities[max_similarity_index] * 100  # Convert to percentage
            closest_sentence = sentences[max_similarity_index]

            # Assign proximity rating based on cosine similarity score
            if max_similarity_score <= 25:
                proximity_rating = "negligible or no similarity"
            elif max_similarity_score <= 50:
                proximity_rating = "very low similarity"
            elif max_similarity_score <= 75:
                proximity_rating = "moderate similarity"
            elif max_similarity_score <= 90:
                proximity_rating = "strong similarity"
            else:
                proximity_rating = "extremely strong similarity"

            results.append((row['filepath'], proximity_rating, closest_sentence, max_similarity_score))

        results_df = pd.DataFrame(results, columns=['file_name', 'proximity_rating', 'closest_sentence', 'cosine_similarity'])
        results_df.sort_values(by='cosine_similarity', ascending=False, inplace=True)
        
        return results_df

# Instantiate TextEmbeddingManager
manager = TextEmbeddingManager()

# Define layout for directory input, embedding generation, and search
st.markdown("### Step 1: Upload Text Files")
uploaded_files = st.file_uploader("Upload text files", type="txt", accept_multiple_files=True)

st.markdown("### Step 2: Generate Sentence Embeddings")
if st.button("Generate Embeddings"):
    if uploaded_files:
        manager.collect_txt_files_data(uploaded_files)
        manager.encode_sentences_sbert()
        st.success("Embeddings generated successfully.")

        # Save embeddings to pickle and create a download button
        pickle_data = manager.save_data_pickle()
        st.download_button(
            label="Download Embeddings as Pickle",
            data=pickle_data,
            file_name=f"text_embeddings_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pickle",
            mime="application/octet-stream"
        )
    else:
        st.error("Please upload text files to generate embeddings.")

# Load data from pickle
st.markdown("### Step 3: Load Existing Embeddings")
uploaded_pickle = st.file_uploader("Load embeddings from a pickle file", type="pickle", help="Load pre-existing embeddings to avoid regenerating them.")
if uploaded_pickle:
    manager.load_data_pickle(uploaded_pickle.read())
    st.success("Pickle data loaded successfully.")

# Search and save closest sentences
st.markdown("### Step 4: Search for Similar Sentences")
target_sentence = st.text_input("Enter a target sentence to find closest matches", help="Enter a sentence to find its closest matches in your dataset.")

if st.button("Find Closest Sentences"):
    if target_sentence and manager.df is not None:
        results_df = manager.find_and_save_closest_sentences(target_sentence)
        st.dataframe(results_df)

        # Provide option to download results
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="closest_sentences.csv",
            mime="text/csv"
        )
    else:
        st.error("Please upload data, generate embeddings, and enter a target sentence.")
