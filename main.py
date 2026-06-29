"""
FastAPI wrapper for the KDP Catalog RAG system.
Runs the RAG pipeline and exposes it as a web API.
"""
from dotenv import load_dotenv
load_dotenv()

import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
import os

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="kdp_catalog")
claude_client = anthropic.Anthropic()

# FastAPI app
app = FastAPI(
    title="KDP Catalog RAG",
    description="Search and get information about Mano's KDP book catalog using AI",
    version="1.0"
)

# Request model (what the user sends)
class QuestionRequest(BaseModel):
    question: str

# Response model (what the API returns)
class RAGResponse(BaseModel):
    question: str
    retrieved_books: list
    answer: str

# Your books
books = [
    {
        "title": "Dash Dragon",
        "description": "A brave little orange dragon with a red cape embarks on an adventure through dark forests, rescues friends, and discovers his courage. For ages 3-6. 32 pages, read-and-color format.",
    },
    {
        "title": "Menopause Wellness Journal",
        "description": "A guided journal for women navigating menopause with practical wellness tips, symptom tracking, and self-care activities. Includes meditations and nutrition advice.",
    },
    {
        "title": "Cassette Dreams",
        "description": "An 80s nostalgia coloring book for adults aged 40+ featuring retro themes including cassette tapes, arcade machines, boom boxes, and vintage video games. A warm trip down memory lane. Price: $14.99.",
    },
    {
        "title": "She Recovers",
        "description": "A 30-day burnout recovery tracker for professional women aged 30-50 to restore energy, rebuild balance, and reclaim their life after exhaustion. Gentle, structured daily layouts. Price: $14.99.",
    },
    {
        "title": "Her Best Month",
        "description": "A 30-day daily tracker for working women aged 28-45 to balance career, family, and wellness without the overwhelm. Includes priority planning, energy tracking, and wellness check-ins. Price: $14.99.",
    },
    {
        "title": "ADHD Planner for Women",
        "description": "A weekly organizer designed specifically for women aged 25-45 with ADHD. Features brain dump sections, habit tracker, and mood check-in to stay focused and organized without the overwhelm. Price: $19.99.",
    },
]

def embed_text(text: str) -> list:
    """Embed text using sentence-transformers."""
    embedding = embedding_model.encode(text, convert_to_numpy=False)
    return embedding.tolist()

def init_catalog():
    """Index books into Chroma (called once on startup)."""
    for book in books:
        title = book["title"]
        description = book["description"]
        embedding = embed_text(description)
        
        collection.add(
            documents=[description],
            ids=[title],
            embeddings=[embedding],
            metadatas=[{"title": title}]
        )

def retrieve_books(query: str, top_k: int = 2) -> list:
    """Retrieve similar books."""
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    retrieved = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "title": metadata["title"],
            "description": doc
        })
    
    return retrieved

def answer_with_rag(question: str) -> dict:
    """Full RAG pipeline."""
    retrieved = retrieve_books(question, top_k=2)
    
    context = "Here are relevant books from the catalog:\n\n"
    for i, book in enumerate(retrieved, 1):
        context += f"{i}. **{book['title']}**\n{book['description']}\n\n"
    
    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system="You are a helpful assistant for a KDP book catalog. Answer questions about books based on the catalog information provided. Be concise and helpful.",
        messages=[
            {
                "role": "user",
                "content": f"""Based on these books:

{context}

Answer this question: {question}"""
            }
        ]
    )
    
    answer = response.content[0].text
    
    return {
        "question": question,
        "retrieved_books": [b["title"] for b in retrieved],
        "answer": answer
    }

# Initialize catalog on startup
@app.on_event("startup")
def startup_event():
    init_catalog()
    print("✓ Catalog indexed and ready")

# The main endpoint
@app.post("/ask", response_model=RAGResponse)
def ask_about_catalog(request: QuestionRequest):
    """
    Ask a question about the KDP catalog.
    Returns the answer plus the books it used to answer.
    """
    result = answer_with_rag(request.question)
    return RAGResponse(**result)

# Health check endpoint (Render uses this to verify the app is alive)
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Root endpoint (nice to have)
@app.get("/")
def root():
    return {
        "message": "KDP Catalog RAG API",
        "docs": "/docs",
        "health": "/health",
        "endpoint": "/ask"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)