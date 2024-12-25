# LocalMind

LocalMind is a Retrieval-Augmented Generation (RAG) application that uses TF-IDF for context retrieval and integrates with the Llama model to provide intelligent answers to user queries based on uploaded PDF documents. This tool is built with Streamlit for an interactive user interface, making it simple to upload documents and ask questions.

## Features

- **PDF Text Extraction**: Upload a PDF document, and LocalMind will extract its text for processing.
- **Text Chunking**: The extracted text is split into manageable chunks for efficient context retrieval.
- **Context Retrieval**: Uses TF-IDF and cosine similarity to identify the most relevant chunks of text based on user queries.
- **Llama Model Integration**: Leverages the power of the Llama model to generate intelligent, context-aware answers.
- **Interactive UI**: A user-friendly interface built with Streamlit for seamless interaction.

## Requirements

- Python 3.8 or higher
- Streamlit
- scikit-learn
- PyPDF2
- numpy
- litellm
- ollama

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ayushroy64/localmind.git
   cd localmind
2. Install requirements:
   ```bash
   pip install -r requirements.txt
3. Ollama:
   You'll need to download ollama (https://ollama.com/download) and run the following command in an administrator (sudo) terminal -
   ```bash
   ollama pull llama3
4. Run the app:
   ```bash
   streamlit run local_RAG.py
