# KDP Catalog RAG

A retrieval-augmented generation (RAG) system for Mano's KDP book catalog. Ask questions about books and get AI-powered answers grounded in the actual catalog.

## Live Demo

**API Endpoint:** `https://kdp-rag.onrender.com`

## How It Works

1. **Retrieve:** User question is embedded and matched against book descriptions
2. **Augment:** Top matching books are added to the prompt
3. **Generate:** Claude synthesizes an answer based on the retrieved books

## Quick Start

### Local Development

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py
```

Visit `http://localhost:8000/docs` for interactive API docs.

### API Usage

**POST `/ask`**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Do you have books for ADHD?"}'
```

Response:
```json
{
  "question": "Do you have books for ADHD?",
  "retrieved_books": ["ADHD Planner for Women"],
  "answer": "Yes! The ADHD Planner for Women is specifically designed..."
}
```

## Architecture

- **Embeddings:** Sentence-transformers (local, free)
- **Vector DB:** Chromadb (in-memory)
- **LLM:** Anthropic Claude API
- **API Framework:** FastAPI
- **Deployment:** Render

## Stack & Learning

Built in Week 3-4 of an AI Engineer ramp:
- Week 1: LLM API tool calling
- Week 2: Embeddings & vector databases
- Week 3: RAG pipeline
- Week 4: **FastAPI deployment** (this project)

## Next Steps

- [ ] Add metadata filtering (category, age range, etc.)
- [ ] Implement prompt caching for faster responses
- [ ] Add user feedback mechanism for retrieval quality
- [ ] Expand catalog and tune chunk strategy
