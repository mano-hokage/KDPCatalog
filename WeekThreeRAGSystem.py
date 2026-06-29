"""
WEEK 3 — RAG System
Retrieve relevant books from your catalog, then ask Claude to answer questions about them.
"""

from dotenv import load_dotenv
load_dotenv()

import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Chroma and Claude
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="kdp_catalog")
claude_client = anthropic.Anthropic()

# Your KDP books (same as Week 2)
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
    """Embed text using local sentence-transformers."""
    embedding = embedding_model.encode(text, convert_to_numpy=False)
    return embedding.tolist()

def index_books():
    """Index all books into Chroma (same as Week 2)."""
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
        print(f"[indexed] {title}")

def retrieve_books(query: str, top_k: int = 3) -> list:
    """
    RETRIEVE step: embed the query and find the most similar books.
    """
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results for passing to Claude
    retrieved_books = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        retrieved_books.append({
            "title": metadata["title"],
            "description": doc
        })
    
    return retrieved_books

def answer_question(question: str) -> dict:
    """
    Full RAG pipeline:
    1. RETRIEVE relevant books
    2. AUGMENT the prompt with those books
    3. GENERATE an answer using Claude
    """
    
    # Step 1: Retrieve
    print(f"\n[query] {question}")
    retrieved = retrieve_books(question, top_k=3)
    
    print(f"[retrieved {len(retrieved)} books]")
    for book in retrieved:
        print(f"  - {book['title']}")
    
    # Step 2: Augment - build the context for Claude
    context = "Here are relevant books from the catalog:\n\n"
    for i, book in enumerate(retrieved, 1):
        context += f"{i}. **{book['title']}**\n{book['description']}\n\n"
    
    # Step 3: Generate - ask Claude to answer based on the context
    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system="You are a helpful assistant for a KDP book catalog. Answer questions about books based on the catalog information provided. If a book doesn't exist in the catalog, say so clearly.",
        messages=[
            {
                "role": "user",
                "content": f"""Based on these books from the catalog:

{context}

Please answer this question: {question}

If the answer isn't in the retrieved books, say so clearly."""
            }
        ]
    )
    
    answer = response.content[0].text
    
    return {
        "question": question,
        "retrieved_books": [b["title"] for b in retrieved],
        "answer": answer
    }

def main():
    # Index the books once
    print("[indexing books...]")
    index_books()
    
    # Test queries
    test_queries = [
        "Do you have any dragon books for young kids?",
        "What books help with women's wellness?",
        "I'm looking for nostalgic coloring books for adults",
        "Do you have anything for ADHD?",
        "Tell me about your children's books",
    ]
    
    for query in test_queries:
        result = answer_question(query)
        print(f"\n[answer] {result['answer']}")
        print("-" * 80)

if __name__ == "__main__":
    main()