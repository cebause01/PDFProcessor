import streamlit as st
import requests
import PyPDF2
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Helper Functions

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() or ""
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    if not text:
        return chunks
    # Split by words to ensure chunks don't cut words in half
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    i = 0
    while i < len(words):
        chunk = words[i:min(i + chunk_size, len(words))]
        chunks.append(" ".join(chunk))
        # Move to the next chunk, accounting for overlap
        i += chunk_size - overlap
        # Ensure 'i' doesn't go out of bounds or negative in edge cases
        if i >= len(words) and len(chunk) < chunk_size: # Avoid empty chunks if at the end
            break
        if i < 0: # Should not happen with positive chunk_size and overlap
            i = 0
    return chunks

# Function to get embeddings from Ollama
# Using Streamlit's cache_data for performance optimization
@st.cache_data(show_spinner="Generating embeddings (this might take a moment)...")
def get_ollama_embeddings(text_items, ollama_api_url, model="nomic-embed-text"):
    embeddings = []
    # Display a progress bar for embedding generation
    progress_bar = st.progress(0, text=f"Generating embeddings for {len(text_items)} text items...")
    
    for i, item in enumerate(text_items):
        try:
            response = requests.post(
                f"{ollama_api_url}/api/embeddings",
                json={"model": model, "prompt": item},
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if embedding:
                embeddings.append(embedding)
            else:
                st.warning(f"Could not get embedding for an item. Skipping.")
                embeddings.append(None) # Append None to maintain index alignment for now
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting embedding for item: {e}")
            embeddings.append(None)
        # Update progress bar
        progress_bar.progress((i + 1) / len(text_items), text=f"Generated {i+1}/{len(text_items)} embeddings.")
    
    progress_bar.empty() # Remove progress bar after completion
    return [e for e in embeddings if e is not None] # Filter out any None values

# Function to perform similarity search
def retrieve_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    if not chunk_embeddings or not query_embedding:
        return []
    
    # Convert lists to numpy arrays for efficient computation
    query_vec = np.array(query_embedding).reshape(1, -1)
    chunk_vecs = np.array(chunk_embeddings)

    # Calculate cosine similarity between the query and all chunks
    # cosine_similarity returns a 2D array, so we take the first row [0]
    similarities = cosine_similarity(query_vec, chunk_vecs)[0]
    
    # Get indices of the top_k most similar chunks
    # argsort returns indices that would sort an array, [::-1] reverses it for descending order
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    
    # Retrieve the actual text chunks based on the sorted indices
    relevant_chunks = [chunks[i] for i in top_k_indices]
    return relevant_chunks

# Function to chat with Ollama
def chat_with_ollama(prompt, ollama_api_url, model="llama3.2"):
    try:
        with st.spinner("Generating response... This may take a moment."):
            response = requests.post(
                f"{ollama_api_url}/api/chat",
                headers={"Content-Type": "text/plain"},  # Ensure correct Content-Type
                data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}),  # Use data for raw payload
                stream=True,  # Enable streaming response
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            # Process the streaming response
            full_answer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():  # Skip empty lines
                    try:
                        partial_response = json.loads(line)
                        content = partial_response.get("message", {}).get("content", "")
                        full_answer += content
                        # Simulate typing by updating the placeholder
                        st.session_state["typing_placeholder"].markdown(f"**Chat:** {full_answer}")
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing partial response: {e}")
                        st.info(f"Raw partial response: {line}")  # Log raw partial response for debugging

            if not full_answer.strip():
                st.warning("Ollama returned an empty response. Please check the model or server status.")
            return full_answer.strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        return None

# Streamlit Application

st.title("Chat with Your Document")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize session state variables for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []  # Stores the chat history
if 'document_chunks' not in st.session_state:
    st.session_state['document_chunks'] = []
if 'document_embeddings' not in st.session_state:
    st.session_state['document_embeddings'] = []
if 'file_processed' not in st.session_state:
    st.session_state['file_processed'] = False
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = ""

# Process the uploaded file if it's new or not yet processed
if uploaded_file and (not st.session_state['file_processed'] or st.session_state['file_name'] != uploaded_file.name):
    st.info(f"Processing PDF '{uploaded_file.name}' and generating embeddings. This may take a moment, especially for large files...")
    
    # Store the file name to check if a new file is uploaded later
    st.session_state['file_name'] = uploaded_file.name

    # Use BytesIO to read the PDF data, enabling PyPDF2 to work with it
    with io.BytesIO(uploaded_file.getvalue()) as pdf_buffer:
        file_content = extract_text_from_pdf(pdf_buffer)

    if file_content.strip():  # Check if any content was extracted
        st.session_state['document_chunks'] = split_text_into_chunks(file_content)

        if st.session_state['document_chunks']:
            # Generate embeddings for each chunk using the specified Ollama embedding model
            ollama_api_url = "http://localhost:11434"
            embedding_model = "nomic-embed-text"
            st.session_state['document_embeddings'] = get_ollama_embeddings(
                st.session_state['document_chunks'], ollama_api_url, embedding_model
            )
            
            if st.session_state['document_embeddings']:
                st.session_state['file_processed'] = True
            else:
                st.error("Failed to generate embeddings. Please ensure Ollama is running and the embedding model is available.")
                st.session_state['file_processed'] = False  # Reset if embedding failed
        else:
            st.warning("No significant text found in the PDF to chunk. It might be an image-only PDF or too short.")
            st.session_state['file_processed'] = False
    else:
        st.error("Could not extract any text from the PDF. It might be an image-only PDF, password-protected, or empty.")
        st.session_state['file_processed'] = False  # Reset if extraction failed

# Chat interface section, only enabled if a file has been successfully processed
if st.session_state['file_processed']:
    # Display conversation history
    st.subheader("Chat")
    for message in st.session_state['conversation_history']:
        if message["role"] == "user":
            # Align user messages to the right
            st.markdown(f"<div style='text-align: right;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            # Align bot responses to the left
            st.markdown(f"<div style='text-align: left;'><strong>Chat:</strong> {message['content']}</div>", unsafe_allow_html=True)

    # Placeholder for simulating typing
    if "typing_placeholder" not in st.session_state:
        st.session_state["typing_placeholder"] = st.empty()

    # Input for user query
    def process_query():
        user_query = st.session_state["user_query_input"]
        if user_query:
            # Add user query to conversation history
            st.session_state['conversation_history'].append({"role": "user", "content": user_query})
            st.session_state["user_query_input"] = ""  # Clear the input field

            if st.session_state['document_embeddings'] and st.session_state['document_chunks']:
                # Get embedding for the user's query
                query_embedding_list = get_ollama_embeddings([user_query], "http://localhost:11434", "nomic-embed-text")
                
                if query_embedding_list:
                    query_embedding = query_embedding_list[0]  # get_ollama_embeddings returns a list
                    
                    # Retrieve the most relevant chunks from the document
                    relevant_chunks = retrieve_relevant_chunks(
                        query_embedding, 
                        st.session_state['document_embeddings'], 
                        st.session_state['document_chunks'], 
                        top_k=3  # Retrieve top 3 most similar chunks
                    )
                    
                    if relevant_chunks:
                        context = "\n\n".join(relevant_chunks)

                        # Construct the final prompt for the LLM, including the retrieved context and conversation history
                        conversation_context = "\n".join(
                            [f"{msg['role']}: {msg['content']}" for msg in st.session_state['conversation_history']]
                        )
                        final_prompt = f"""Using ONLY the following context, answer the question. If the answer is not directly available in the context, state that you don't have enough information from the provided document.

                        Context:
                        {context}

                        Conversation History:
                        {conversation_context}

                        Question: {user_query}

                        Answer:
                        """
                        # Get the answer from Ollama using the LLM model
                        answer = chat_with_ollama(final_prompt, "http://localhost:11434", "llama3.2")
                        if answer:
                            # Add bot response to conversation history
                            st.session_state['conversation_history'].append({"role": "assistant", "content": answer})
                        else:
                            st.warning("Could not generate an answer from Ollama. Please check Ollama server status or the model selected.")
                    else:
                        st.warning("No relevant information found in the document for your query.")
                else:
                    st.error("Failed to generate an embedding for your query. Please try again.")
            else:
                st.warning("Document processing not complete. Please wait or re-upload the PDF.")

    st.text_input("Ask a question about the uploaded document:", key="user_query_input", on_change=process_query)
else:
    st.info("Upload a PDF file in the sidebar to begin chatting with your document.")

