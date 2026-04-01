# Project Context: UdaPlay AI Research Agent

## Overview
**udaplay-ai-research-agent** is a stateful AI assistant designed to answer natural language questions about video games for executives, analysts, and gamers.

The project implements a two-tier retrieval system using a local RAG pipeline (ChromaDB) as the primary knowledge base, with an LLM-evaluated fallback to web search (Tavily API). The agent is built using LangGraph to maintain conversational memory and state.

## Architecture & Tech Stack
*   **Core Logic:** `langgraph`, `langchain`, `langchain-openai`
*   **Vector Database:** `chromadb` (Local, persistent storage in `vector_db/`)
*   **Web Search API:** `tavily-python`
*   **Data Models:** `pydantic` (for structured LLM outputs and evaluation)
*   **Embeddings & LLMs:** OpenAI (`text-embedding-3-small` and `gpt-4o-mini`)

## Directory Layout & Key Components
*   **`src/database.py`**: Handles recursive data ingestion from `data/raw/`, formats JSON/dict records into textual representations, sanitizes metadata, and manages the ChromaDB collections and semantic search.
*   **`src/tools.py`**: Contains the core action nodes for the agent:
    *   `retrieve_game`: Queries the local ChromaDB.
    *   `evaluate_retrieval`: Uses `gpt-4o-mini` to grade if retrieved context is sufficient.
    *   `game_web_search`: Uses Tavily for fallback web queries.
    *   `persist_web_search_results`: Hashes URLs and injects new web findings back into ChromaDB for long-term memory.
*   **`src/agent.py`**: Defines the LangGraph state machine (`UdaPlayAgent`), managing conditional routing (`should_web_search`) and enforcing structured JSON + natural language responses.
*   **`notebooks/`**:
    *   `Udaplay_01_solution_project.ipynb`: Demonstrates data loading and ChromaDB setup.
    *   `Udaplay_02_solution_project.ipynb`: Demonstrates agent execution, including RAG queries, web fallback, and stateful multi-turn conversations.
*   **`data/raw/`**: Contains raw source data, including individual JSON files under `games/` and an aggregated `video-games-ranking.yaml` file.
