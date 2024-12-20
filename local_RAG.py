import streamlit as st
from litellm import completion
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import numpy as np


# Utility function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to split text into smaller chunks
def split_text_into_chunks(text, max_chunk_size=500):
    chunks = []
    current_chunk = ""

    for paragraph in text.split("\n"):
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + " "
    if current_chunk:  # Append the last chunk if not empty
        chunks.append(current_chunk.strip())

    return chunks


# Function to retrieve relevant context using TF-IDF and cosine similarity
def retrieve_relevant_context(query, chunks, top_n=3):
    vectorizer = TfidfVectorizer()
    all_texts = [query] + chunks
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    query_vector = tfidf_matrix[0]
    chunk_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    relevant_chunks = [chunks[i] for i in top_indices]
    return " ".join(relevant_chunks)


# Set up the Streamlit app
st.title("Llama-Powered RAG (Retrieval-Augmented Generation)")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF for context", type=["pdf"])

# Process the PDF if uploaded
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("PDF processed successfully!")
    text_chunks = split_text_into_chunks(pdf_text)
    st.write(f"Extracted {len(text_chunks)} chunks from the PDF.")

# Input for user query
user_query = st.text_input("Ask a question:")

if st.button("Generate Answer"):
    if not uploaded_file:
        st.warning("Please upload a PDF file to use as context.")
    elif not user_query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            # Retrieve relevant context
            context = retrieve_relevant_context(user_query, text_chunks)

            # Use Llama to generate a response with context
            prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"
            response = completion(model="ollama/llama3", messages=[{"role": "user", "content": prompt}])

            # Display the response
            st.subheader("Response from Llama:")
            st.write(response.choices[0].message.content)

# Sidebar Information
st.sidebar.write("### Instructions:")
st.sidebar.write("- Upload a PDF document for context.")
st.sidebar.write("- Enter your query.")
st.sidebar.write("- Click 'Generate Answer' to get a response.")
