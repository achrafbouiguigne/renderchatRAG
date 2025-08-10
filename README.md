# RAG Chatbot - Ruthless Motivational Coach 🔥

A Flask-based RAG (Retrieval Augmented Generation) chatbot that provides savage motivational coaching using content from a PDF document. Built with FAISS for vector search and Google's Gemini AI for response generation.

## Features

- 📚 **PDF-based Knowledge Base**: Processes and chunks PDF content for retrieval  
- 🔍 **Vector Search**: Uses FAISS and TF-IDF for semantic document retrieval  
- 🧠 **Conversation Memory**: Maintains context across interactions  
- 💪 **Ruthless Coaching Style**: No-nonsense motivational responses  
- 🌐 **REST API**: Clean JSON API for frontend integration  
- ☁️ **Cloud Ready**: Optimized for Render deployment  

## Tech Stack

- **Backend**: Flask, Python 3.x  
- **Vector Store**: FAISS (Facebook AI Similarity Search)  
- **PDF Processing**: PyMuPDF (fitz)  
- **ML**: scikit-learn (TF-IDF), NumPy  
- **AI**: Google Gemini 2.0 Flash  
- **Deployment**: Render (gunicorn)  

## Project Structure

rag-chatbot/
├── app.py            # Main Flask application
├── requirements.txt  # Python dependencies
├── ruthless.pdf      # Knowledge base PDF
├── README.md         # This file
└── .gitignore        # Git ignore rules

## Local Development

### Prerequisites

- Python 3.8+  
- pip package manager  
- Google Gemini API key  

### Installation

1. **Clone the repository**  
   ```bash
   git clone  https://github.com/achrafbouiguigne/renderchatRAG.git
   cd RAG
   ````

2. **Install dependencies**
 ```bash
    pip install -r requirements.txt
   ````


3. *Add your PDF**  
   ```bash
   Place your knowledge base PDF as ruthless.pdf in the root directory
   ````

4. **Set up environment variables (optional for local)**
 ```bash
export GEMINI_API_KEY="your_api_key_here"
   ````


4. **Run the application


**
 ```bash
python app.py
   ````
