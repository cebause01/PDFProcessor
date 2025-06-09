import streamlit as st
import requests
import PyPDF2
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json


def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "".join([page.extract_text() or "" for page in reader.pages])

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

@st.cache_data(show_spinner="Embeddings...")
def get_ollama_embeddings(text_items, ollama_api_url, model="nomic-embed-text"):
    embeddings = []
    progress = st.progress(0)
    for i, item in enumerate(text_items):
        try:
            res = requests.post(f"{ollama_api_url}/api/embeddings", json={"model": model, "prompt": item})
            res.raise_for_status()
            embedding = res.json().get("embedding")
            embeddings.append(embedding if embedding else None)
        except:
            embeddings.append(None)
        progress.progress((i + 1) / len(text_items))
    return [e for e in embeddings if e is not None]

def retrieve_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    query_vec = np.array(query_embedding).reshape(1, -1)
    chunk_vecs = np.array(chunk_embeddings)
    similarities = cosine_similarity(query_vec, chunk_vecs)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# ---------- Streamlit App Starts ----------

st.set_page_config(page_title="Document Chat", layout="centered")
st.title("📄 Chat with Your PDF Document")

# Sidebar Upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Session Initialization
for key in ['conversation_history', 'document_chunks', 'document_embeddings', 'file_processed', 'file_name']:
    if key not in st.session_state:
        st.session_state[key] = [] if 'history' in key or 'chunks' in key or 'embeddings' in key else False if 'processed' in key else ""

# File Processing
if uploaded_file and (not st.session_state['file_processed'] or st.session_state['file_name'] != uploaded_file.name):
    st.session_state['file_name'] = uploaded_file.name
    st.info(f"Processing: {uploaded_file.name}")
    with io.BytesIO(uploaded_file.getvalue()) as pdf_buffer:
        text = extract_text_from_pdf(pdf_buffer)
    if text.strip():
        chunks = split_text_into_chunks(text)
        embeddings = get_ollama_embeddings(chunks, "http://localhost:11434")
        if embeddings:
            st.session_state['document_chunks'] = chunks
            st.session_state['document_embeddings'] = embeddings
            st.session_state['file_processed'] = True
        else:
            st.error("Failed to get embeddings.")
    else:
        st.error("No readable text found in the PDF.")

# Chat Interface
if st.session_state['file_processed']:
    chat_container = st.empty()
    input_container = st.container()

    def render_chat():
        with chat_container:
            chat_html = """
<style>
    .chat-bubble {
        max-width: 80%;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 15px;
        font-size: 15px;
        line-height: 1.5;
        color: black;  /* Default black font for all bubbles */
        word-wrap: break-word;  /* Ensure text wraps within the bubble */
        overflow-wrap: break-word;  /* Handle long words or URLs */
    }
    .user {
        background-color: #DCF8C6;
        margin-left: auto;
        text-align: right;
        color: black;
    }
    .assistant {
        background-color: #F1F0F0;
        margin-right: auto;
        text-align: left;
        color: black;
    }
    #chat-scroll {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.3);
    }
</style>
<div id="chat-scroll">
"""

            for msg in st.session_state['conversation_history']:
                role = msg["role"]
                content = msg["content"].replace("\n", "<br>")
                css_class = "user" if role == "user" else "assistant"
                chat_html += f'<div class="chat-bubble {css_class}">{content}</div>'
            chat_html += """
</div>
<script>
    const chatBox = document.getElementById("chat-scroll");
    if (chatBox) {
        chatBox.scrollBottom = chatBox.scrollHeight;
    }
</script>
"""
            st.markdown(chat_html, unsafe_allow_html=True)

    def process_query():
        user_query = st.session_state["user_query_input"]
        if not user_query:
            return

        # Clear input immediately
        st.session_state["user_query_input"] = ""

        # Append user message
        st.session_state['conversation_history'].append({"role": "user", "content": user_query})
        render_chat()

        # Get embedding
        embedding = get_ollama_embeddings([user_query], "http://localhost:11434")
        if not embedding:
            st.error("Failed to embed query.")
            return

        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(
            embedding[0],
            st.session_state['document_embeddings'],
            st.session_state['document_chunks']
        )

        if not relevant_chunks:
            st.warning("No relevant information found.")
            return

        context = "\n\n".join(relevant_chunks)
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state['conversation_history']])
        prompt = f"""Using ONLY the following context, answer the question. If the answer is not directly available in the context, say you don't know.

Context:
{context}

Conversation History:
{history}

Question: {user_query}

Answer:"""

        # Add assistant placeholder
        st.session_state['conversation_history'].append({"role": "assistant", "content": ""})
        assistant_index = len(st.session_state['conversation_history']) - 1

        # Stream response
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                headers={"Content-Type": "text/plain"},
                data=json.dumps({"model": "llama3.2", "messages": [{"role": "user", "content": prompt}]}),
                stream=True
            )
            response.raise_for_status()
            full_answer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():
                    try:
                        partial = json.loads(line)
                        content = partial.get("message", {}).get("content", "")
                        full_answer += content
                        st.session_state['conversation_history'][assistant_index]["content"] = full_answer
                        render_chat()
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.RequestException as e:
            st.error(f"Error from Ollama: {e}")

    # Render chat and input
    render_chat()
    with input_container:
        st.text_input("Type your question:", key="user_query_input", on_change=process_query)
else:
    st.info("Upload a PDF to start chatting.")
