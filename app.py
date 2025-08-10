from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os

# === Config ===
# ONLY CHANGE: Use environment variable for API key in production
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCAi1XPo_dBVxTjTln7LVJVMvBgRd1Qzgk")

PDF_PATH = "ruthless.pdf"
MAX_MEMORY_LENGTH = 3

# === App Setup ===
app = Flask(__name__)
CORS(app)

# === Conversation Memory ===
conversation_memory = []

def update_conversation_memory(user_query, bot_response):
    conversation_memory.append((user_query, bot_response))
    if len(conversation_memory) > MAX_MEMORY_LENGTH:
        conversation_memory.pop(0)

# === Load PDF & Chunk ===
def load_pdf_chunks(pdf_path, chunk_size=500, overlap=50):
    doc = fitz.open(pdf_path)
    full_text = "".join([page.get_text() for page in doc])
        
    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk = full_text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

# === Build Vector Store ===
pdf_chunks = load_pdf_chunks(PDF_PATH)
docs = [Document(page_content=chunk) for chunk in pdf_chunks]
texts = [doc.page_content for doc in docs]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts).toarray().astype('float32')

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

faiss_store = FAISS(
    embedding_function=None,
    index=index,
    docstore=docs,
    index_to_docstore_id=lambda i: str(i)
)

# === Retrieval Function ===
def retrieve(query, vectorizer, faiss_store, k=3):
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    _, indices = faiss_store.index.search(query_vector, k)
    return [faiss_store.docstore[i] for i in indices[0]]

# === Generate Gemini Response ===
def generate_ruthless_response(query, relevant_docs):
    context = "\n".join([doc.page_content for doc in relevant_docs])
    memory = "\n".join([f"User: {q}\nBot: {a}" for q, a in conversation_memory])

    prompt = f"""
You're a savage motivational chatbot. No sugarcoating.

Recent conversation: {memory}

Context: {context}

User Question: {query}

Respond like a badass life coach:
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    headers = { "Content-Type": "application/json" }
    body = { "contents": [ { "parts": [ { "text": prompt } ] } ] }

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# === Flask Route ===
@app.route('/api/query', methods=['POST'])
def query_rag():
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({'error': 'Query missing'}), 400

    relevant_docs = retrieve(user_query, vectorizer, faiss_store)
    answer = generate_ruthless_response(user_query, relevant_docs)
    update_conversation_memory(user_query, answer)
    return jsonify({'response': answer})

# === Run Server ===
if __name__ == '__main__':
    # ONLY CHANGE: Use PORT environment variable for Render
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 2000)))