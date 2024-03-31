# Text Embedding Manager

The Text Embedding Manager is a tool designed for attorneys, activists, and anyone needing to analyze and understand large corpuses of textual data efficiently. By leveraging advanced machine learning techniques, this application allows users to search for sentences semantically similar to a given target sentence-- allowing fast, automated sifting through extensive collections of documents such as body camera transcripts, jail call transcripts, and other large textual datasets.

## Features

- **Text Collection:** Automatically collects and aggregates text data from .txt files within a user-specified directory.
- **Sentence Embedding Generation:** Uses a pretrained Sentence Transformer model ('all-MiniLM-L6-v2') to generate sentence embeddings for nuanced understanding and analysis of textual content.
- **Data Persistence:** Allows users to save and load processed data in a pickle format for easy data management and analysis continuation.
- **Semantic Search:** Provides the capability to find and save the closest semantic matches to a given search phrase within the collected data, facilitating discovery of relevant information quickly. The results are saved to a CSV file, sorted by the filepaths based on the textual content's semantic proximity to the target sentence.
- **GUI Interface:** Features a user-friendly graphical interface that simplifies the management of text embeddings and the search process, making advanced data analysis accessible to non-technical users.

## Installation

Before running the application, ensure you have Python installed on your system. The application requires Python 3.6 or newer. You will also need to install the required dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/ianduke25/Text_Embedding_Manager
   ```
2. Navigate to the application directory:
   ```bash
   cd Text_Embedding_Manager
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the Text Embedding Manager, run the following command in your terminal:

```bash
python text_embedding_manager.py
```

### Initial Setup

1. **Select Directory**: Use the "Select Directory" button to choose the directory containing your .txt files.
2. **Generate Embeddings**: Click "Generate Embeddings" to process the text files and generate embeddings.

### Data Management

- **Load Data (Pickle)**: To continue working with previously processed data, use the "Load Data" button to load your data from a pickle file.

### Searching and Analysis

1. Enter a search phrase in the "Search Phrase" field.
2. Click "Search and Save Results" to find the closest matches within your data. The results will be saved to a CSV file for further analysis.
