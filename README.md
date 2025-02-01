# ai-bootcamp.basic_rag

A basic implementation of Retrieval-Augmented Generation (RAG) for AI bootcamp.

## Project Overview
This project implements a basic RAG pipeline using:
- OpenAI for LLM queries
- Vector database for document storage and retrieval
- Text embedding for semantic search

## Key Components
- `llm.py`: Handles LLM queries and responses
- `embeds.py`: Manages text embeddings and vector operations
- `vectordb.py`: Implements a simple vector database
- `rag.py`: Main RAG pipeline implementation

## Environment Setup

1. Create a virtual environment
```bash
python -m venv venv
```

2. Activate it:
- Windows:
```bash
.\venv\Scripts\activate
```
- MacOS/Linux:
```bash
source ./venv/bin/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with the following variables:
```env
API_KEY=your_openai_api_key
LLM_MODEL=your_preferred_model
```

## Usage

1. Run the RAG pipeline:
```python
python rag.py
```

2. Query the LLM interactively:
```python
python llm.py
```
