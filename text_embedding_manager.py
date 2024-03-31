from datetime import datetime
import numpy as np
import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import re
from threading import Thread
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from tqdm import tqdm

# Class containing functions neeeded to create sentence embeddings
class TextEmbeddingManager:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.df = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    # Comb through directory and grab .txt files
    def collect_txt_files_data(self):
        filepaths, contents = [], []
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if file.endswith('.txt'):
                    filepath = os.path.join(root, file)
                    filepaths.append(filepath)

                    with open(filepath, 'r', encoding='utf-8') as f:
                        contents.append(f.read())

        self.df = pd.DataFrame({
            'filepath': filepaths,
            'transcript': contents
        })

    # Use MiniLM to encode all sentences in .txt files
    def encode_sentences_sbert(self):
        if self.df is None:
            self.collect_txt_files_data()

        self.df['embeddings'] = None
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            text = str(row['transcript'])
            sentences = re.split(r'[.?]+', text)
            sentences = [sentence.strip()
                         for sentence in sentences if sentence.strip()]
            doc_embeddings = self.model.encode(sentences)
            self.df.at[index, 'embeddings'] = [embedding.tolist()
                                               for embedding in doc_embeddings]

    # Save embeddings as a pickle file
    def save_data_pickle(self, pickle_path):
        if self.df is not None:
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.df, f)

    # Load filepaths, text, and embedding lists from pickle
    def load_data_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.df = pickle.load(f)

    # Calculate cosine similarity to find closest semantic match to target
    # sentence
    def find_and_save_closest_sentences(self, target_sentence, output_csv):
        if self.df is None or 'embeddings' not in self.df.columns:
            print("Embeddings not generated. \
                  Please generate embeddings first.")
            return

        target_embedding = self.model.encode([target_sentence])[0]

        results = []
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            sentence_embeddings = np.array(row['embeddings'])
            sentences = re.split(r'[.?\n]+', row['transcript'])
            sentences = [sentence.strip()
                         for sentence in sentences if sentence.strip()]

            if not sentences:
                continue

            # Calcualate similarity score
            similarities = [np.dot(target_embedding, embedding) /
                            (np.linalg.norm(target_embedding) *
                             np.linalg.norm(embedding)) for
                            embedding in sentence_embeddings]
            max_similarity_index = np.argmax(similarities)
            # Isolate closest sentence by locating highest similarity score
            max_similarity_score = similarities[max_similarity_index]
            closest_sentence = sentences[max_similarity_index]

            results.append(
                (row['filepath'],
                 closest_sentence,
                 max_similarity_score))

        results_df = pd.DataFrame(
            results,
            columns=[
                'filepath',
                'closest_sentence',
                'cosine_similarity'])
        # Sort by cosine similarity score in descending order
        results_df.sort_values(
            by='cosine_similarity',
            ascending=False,
            inplace=True)
        # Write results to user-friendly csv
        results_df.to_csv(output_csv, index=False)


# Class containing GUI functions
class TextEmbeddingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Text Embedding Manager")

        self.manager = None

        # Define buttons
        ttk.Button(
            self.master,
            text="Select Directory",
            command=self.select_directory).pack(
            pady=5)
        ttk.Button(
            self.master,
            text="Generate Embeddings",
            command=self.generate_embeddings).pack(
            pady=5)
        ttk.Button(
            self.master,
            text="Load Data (Pickle)",
            command=self.load_data).pack(
            pady=5)

        self.search_frame = ttk.Frame(self.master)
        self.search_frame.pack(pady=5)
        ttk.Label(
            self.search_frame,
            text="Search Phrase:").grid(
            row=0,
            column=0)
        self.search_entry = ttk.Entry(self.search_frame)
        self.search_entry.grid(row=0, column=1)
        ttk.Button(
            self.search_frame,
            text="Search and Save Results",
            command=self.search_phrase).grid(
            row=1,
            column=0,
            columnspan=2)

    # Trigger definition of directory path upon "Select Directory" button click
    def select_directory(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.manager = TextEmbeddingManager(directory_path)
            messagebox.showinfo(
                "Directory Selected",
                "Directory has been selected.\
                      You can now generate embeddings.")

    # Trigger embedding generation upon "Generate Embeddings" button click
    def generate_embeddings(self):
        if self.manager:
            Thread(target=self._generate_embeddings_thread).start()
        else:
            messagebox.showerror("Error", "Please select a directory first.")

    def _generate_embeddings_thread(self):  # Save as pickle
        try:
            self.manager.collect_txt_files_data()
            self.manager.encode_sentences_sbert()
            # Generate a filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pickle_filename = f"text_embeddings_{timestamp}.pickle"
            # Save to the current working directory
            pickle_path = os.path.join(os.getcwd(), pickle_filename)
            self.manager.save_data_pickle(pickle_path)
            messagebox.showinfo(
                "Success",
                "Embeddings have been generated" +
                f" and saved as '{pickle_filename}' successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Load pickle file upon "Load Data (Pickle)" button click
    def load_data(self):
        pickle_path = filedialog.askopenfilename(
            filetypes=[("Pickle Files", "*.pickle")])
        if pickle_path:
            self.manager = TextEmbeddingManager('')
            self.manager.load_data_pickle(pickle_path)
            messagebox.showinfo(
                "Data Loaded",
                "Data has been loaded successfully from the pickle file.")

    # Allow user to search for target phrase. Save to .csv
    def search_phrase(self):
        if self.manager and 'embeddings' in self.manager.df.columns:
            target_sentence = self.search_entry.get()
            if target_sentence:
                output_csv = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV Files", "*.csv")])
                if output_csv:
                    Thread(
                        target=lambda:
                        self.manager.find_and_save_closest_sentences(
                            target_sentence, output_csv)).start()
                    messagebox.showinfo(
                        "Success", "The search is complete.\
                         Results have been saved to the CSV file.")
            else:
                messagebox.showerror("Error", "Please enter a search phrase.")
        else:
            messagebox.showerror(
                "Error", "Please load data or generate embeddings first.")


if __name__ == "__main__":
    root = tk.Tk()
    app = TextEmbeddingGUI(root)
    root.mainloop()