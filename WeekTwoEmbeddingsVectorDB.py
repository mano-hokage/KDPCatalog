"""
WEEK 2 — Embeddings + Vector DB (Chroma)
Load your KDP book descriptions, embed them, and search.
"""


from dotenv import load_dotenv
load_dotenv()

import chromadb
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str) -> list:
    """
    Embed text using a local sentence-transformers model.
    Free, runs locally, no API key needed.
    """
    embedding = embedding_model.encode(text, convert_to_numpy=False)
    return embedding.tolist()

# Initialize Chroma client (runs in memory for now, no database to set up)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="kdp_catalog")

# Your actual KDP book descriptions
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

# ❌ DELETE THE SECOND embed_text() FUNCTION (lines ~50-64 in your version)
# It's the placeholder that's breaking everything.
# Keep only the first one above.

def index_books():
    """
    Embed each book description and store in Chroma.
    """
    for book in books:
        title = book["title"]
        description = book["description"]
        
        # Embed the description
        embedding = embed_text(description)
        
        # Add to Chroma collection
        # Chroma needs: documents (the text), ids (unique identifier), embeddings (the vectors)
        collection.add(
            documents=[description],
            ids=[title],
            embeddings=[embedding],  # Chroma stores the embedding
            metadatas=[{"title": title}]
        )
        print(f"[indexed] {title}")

def search(query: str, top_k: int = 2):
    """
    Embed the user's query and find the most similar books.
    """
    query_embedding = embed_text(query)
    
    # Chroma's similarity_search returns the closest vectors
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print(f"\n[query] '{query}'")
    for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
        print(f"  #{i+1} (similarity: {1 - distance:.3f}): {doc[:100]}...")

def main():
    # Index your books
    index_books()
    
    # Test queries
    test_queries = [
        # "Dragon stories for kids",
        # "Journal for women's health",
        # "Children's coloring books",
        # "Adventures with brave characters",
        "tell me about your book about a red cape"
    ]
    
    for query in test_queries:
        search(query, top_k=2)

if __name__ == "__main__":
    main()