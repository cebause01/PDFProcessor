import os
import kdbai_client as kdbai
import requests
from pdf2image import convert_from_bytes
import base64
import io
import re
import ollama
import json
import numpy as np
import pandas as pd

# We'll connect to KDB.AI to store our chunk embeddings
KDBAI_ENDPOINT = "https://cloud.kdb.ai/instance/6hqptbblu8"  # Replace with your endpoint
KDBAI_API_KEY = "9b7df475d4-ZH32FB1VGQtHFny9ZmChtEVl/pp5R56uEeosKvrRXLk8PKU+DiDBDL9wkPYRbUk/Vs/i6VgVukQYzW8+"  # Replace with your API key
session = kdbai.Session(endpoint=KDBAI_ENDPOINT, api_key=KDBAI_API_KEY)
db = session.database('default')
print("Connected to KDB.AI:", db)

# SNIPPET 2: Define KDB.AI table schema
VECTOR_DIM = 384  # We'll use all-MiniLM-L6-v2 for embeddings (but ollama will do it)

schema = [
    {"name": "id", "type": "str"},
    {"name": "text", "type": "str"},
    {"name": "vectors", "type": "float32s"}
]
# Build a simple L2 distance index
index = [
    {
        "name": "flat_index",
        "type": "flat",
        "column": "vectors",
        "params": {"dims": VECTOR_DIM, "metric": "L2"}
    }
]
table_name = "pdf_chunks"
try:
    db.table(table_name).drop()
except kdbai.KDBAIException:
    pass
table = db.create_table(table_name, schema=schema, indexes=index)
print(f"Table '{table_name}' created.")

pdf_url = "https://arxiv.org/pdf/2404.08865"  # example PDF
resp = requests.get(pdf_url)
pdf_data = resp.content
pages = convert_from_bytes(pdf_data)
print(f"Converted {len(pages)} PDF pages to images.")
# We'll encode the images as base64 for easy sending to Ollama
images_b64 = {}
for i, page in enumerate(pages, start=1):
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    image_data = buffer.getvalue()
    b64_str = base64.b64encode(image_data).decode("utf-8")
    images_b64[i] = b64_str

CHUNKING_PROMPT = """\
OCR the following page into Markdown. Tables should be formatted as HTML.
Do not surround your output with triple backticks.
Chunk the document into sections of roughly 250 - 1000 words.
Surround each chunk with <chunk> and </chunk> tags.
Preserve as much content as possible, including headings, tables, etc.
"""

EMBEDDING_PROMPT = """
Return the embedding of the following text as a json list of floats. Do not include any other text.
Text:
"""

OLLAMA_API_URL = "http://localhost:11434" #Replace with your ollama api url, example: http://your_ollama_host:11434
def process_page(page_num, image_b64, model="llama3.2"):  # Using llava for image processing
    payload = f"{CHUNKING_PROMPT}\n{image_b64}"
    try:
        response = requests.post(f"{OLLAMA_API_URL}/api/generate", json={"model": model, "messages": [{"role": "user", "content": payload}]})
        response.raise_for_status()  # Raise an exception for bad status codes
        text_out = response.json()['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error processing page {page_num}: {e}")
        return []

    chunks = re.findall(r"<chunk>(.*?)</chunk>", text_out, re.DOTALL)
    if not chunks:
        chunks = text_out.split("\n\n")

    results = []
    for idx, chunk_txt in enumerate(chunks):
        results.append({
            "id": f"page_{page_num}_chunk_{idx}",
            "text": chunk_txt.strip()
        })
    return results

def get_embedding(text, model="nomic-embed-text"): #use nomic-embed-text for embeddings
    payload = f"{EMBEDDING_PROMPT}{text}"
    try:
        response = requests.post(f"{OLLAMA_API_URL}/api/embed", json={"model": model, "messages": [{"role": "user", "content": payload}]})
        response.raise_for_status()
        embedding_str = response.json()['message']['content'].strip()
        embedding = json.loads(embedding_str)
        return np.array(embedding, dtype=np.float32)
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding: {e}")
        return None

all_chunks = []
for i, b64_str in images_b64.items():
    page_chunks = process_page(i, b64_str)
    all_chunks.extend(page_chunks)
print(f"Total extracted chunks: {len(all_chunks)}")

row_list = []
for ch_data in all_chunks:
    embedding = get_embedding(ch_data["text"])
    if embedding is not None:
        row_list.append({
            "id": ch_data["id"],
            "text": ch_data["text"],
            "vectors": embedding.tolist()
        })

df = pd.DataFrame(row_list)
table.insert(df)
print(f"Inserted {len(df)} chunks into '{table_name}'.")

user_query = "How does this paper handle multi-column text?"
qvec = get_embedding(user_query)

if qvec is not None:
    search_results = table.search(vectors={"flat_index": [qvec]}, n=3)
    retrieved_chunks = search_results[0]["text"].tolist()
    context_for_llm = "\n\n".join(retrieved_chunks)
    print("Retrieved chunks:\n", context_for_llm)

    final_prompt = f"""Use the following context to answer the question:
    Context:
    {context_for_llm}
    Question: {user_query}
    Answer:
    """
    try:
        response = requests.post(f"{OLLAMA_API_URL}/api/chat", json={"model": "llama2", "messages": [{"role": "user", "content": final_prompt}]})
        response.raise_for_status()
        print("\n=== Ollama's final answer ===")
        print(response.json()['message']['content'])
    except requests.exceptions.RequestException as e:
        print(f"Error generating answer: {e}")
else:
    print("Could not generate query embedding.")