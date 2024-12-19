import streamlit as st
from litellm import completion
import PyPDF2
import faiss
import numpy as np

# App Title
st.title("RAG with Multiple Ollama Models")

# Sidebar Information
st.sidebar.write("Upload a PDF and query multiple Ollama models for their responses!")

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to embed text chunks
def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = completion(model="ollama/llama3", messages=[
            {"role": "system", "content": "Generate an embedding for the text."},
            {"role": "user", "content": chunk}
        ])
        embedding = [float(x) for x in response["embedding"]]
        embeddings.append(embedding)
    return np.array(embeddings, dtype=np.float32)

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    st.write("Processing the uploaded PDF...")
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Split text into chunks
    chunks = split_text(pdf_text)
    st.write(f"Extracted {len(chunks)} chunks from the PDF.")

    # Generate embeddings and build FAISS index
    st.write("Generating embeddings and creating FAISS index...")
    embeddings = generate_embeddings(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.success("FAISS index created! Enter your query below.")

    # Query Input
    user_query = st.text_input("Enter your query:")
    if user_query and st.button("Send"):
        # Embed the user query
        query_response = completion(model="ollama/llama3", messages=[
            {"role": "system", "content": "Generate an embedding for the query."},
            {"role": "user", "content": user_query}
        ])
        query_embedding = np.array([float(x) for x in query_response["embedding"]], dtype=np.float32)

        # Retrieve relevant chunks
        distances, indices = index.search(np.array([query_embedding]), k=5)
        relevant_chunks = [chunks[i] for i in indices[0]]

        # Display retrieved chunks
        st.write("Top Relevant Chunks:")
        for i, chunk in enumerate(relevant_chunks):
            st.write(f"**Chunk {i + 1}:**")
            st.write(chunk)

        # Combine chunks into context
        context = "\n\n".join(relevant_chunks)

        # Query Multiple Models
        st.write("Generating responses from multiple models...")
        col1, col2 = st.columns(2)

        # First Model
        with col1:
            st.subheader("llama3")
            try:
                response = completion(model="ollama/llama3", messages=[
                    {"role": "system", "content": "Answer the query based on the following context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuery: {user_query}"}
                ])
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {str(e)}")

        # Second Model
        with col2:
            st.subheader("gemma2:2b")
            try:
                response = completion(model="ollama/gemma2:2b", messages=[
                    {"role": "system", "content": "Answer the query based on the following context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuery: {user_query}"}
                ])
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {str(e)}")

        st.success("Responses generated!")
