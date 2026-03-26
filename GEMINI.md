# Project Context UdaPlay: AI Research Agent - Gemini Context Document

## Overview
**udaplay-ai-research-agent** is a repository for a Udacity exercise project. The project involves an AI assistant called **UdaPlay**, designed to answer natural language questions from executives, analysts, and gamers.

## Directory Layout
*   `README.md`: The main documentation file for the project, currently filled with template placeholders for dependencies, installation steps, and testing information.
*   `LICENSE`: The license file governing the repository's usage.
*   `GEMINI.md`: This file, providing contextual information for the Gemini CLI agent.
*   `src`: code
*   notebooks: python playbooks

## Usage

## Project Overview
* **Role:** AI Engineer at a gaming analytics company.
* **Product:** UdaPlay, a stateful AI Research Agent designed to answer natural language questions about video games (titles, release dates, platforms, publishers).
* **Core Mechanism:** A two-tier retrieval system using local RAG (ChromaDB) as the primary source, with an evaluated fallback to web search (Tavily API).
* **Key Capabilities:**
  * Maintains conversation state and memory.
  * Evaluates retrieval confidence to prevent hallucinations.
  * Persists discovered web information into long-term memory.
  * Outputs clear, well-cited, and structured answers.

## Tech Stack & Environment
* **Core Libraries:** `chromadb>=1.0.4`, `openai>=1.73.0`, `pydantic>=2.11.3`, `python-dotenv>=1.1.0`, `tavily-python>=0.5.4`, `langgraph`, `langchain`.
* **IDE Setup:** VS Code with a dedicated Python profile.
* **Required Extensions:** `ms-python.python`, `ms-python.vscode-pylance`, `charliermarsh.ruff`, `njpwerner.autodocstring`, `ms-python.isort`, `ms-toolsai.jupyter`, `qwtel.sqlite-viewer`.

## Repository Structure
* **`data/raw/`**: Source JSON files containing game data.
* **`vector_db/`**: Persistent local storage for ChromaDB.
* **`notebooks/`**: 
  * `Udaplay_01_solution_project.ipynb` (Data ingestion & DB setup).
  * `Udaplay_02_solution_project.ipynb` (Agent execution & evaluation).
* **`src/`**: 
  * `database.py` (ChromaDB management, ingestion, deduplication).
  * `tools.py` (Retrieval, evaluation, and web search tools).
  * `agent.py` (LangGraph state machine and memory management).
* **`.vscode/extensions.json`**: Enforces the required IDE extensions.
* **`.env.tpl`**: 1Password template for secret injection.

## v1 Implementation Focus Areas & Review Criteria
* **State & Memory Management (`src/agent.py`):** Evaluate LangGraph's context handling to prevent token overflow and verify the conditional web search routing logic.
* **LLM Evaluation Prompting (`src/tools.py`):** Review the `evaluate_retrieval` prompt to balance strict hallucination prevention with minimizing unnecessary web searches.
* **Data Ingestion & Deduplication (`src/database.py`):** Validate URL hashing for deduplicating stored web results and ensure ChromaDB metadata is properly sanitized (strings, ints, floats, bools only).
* **Graceful API Degradation:** Confirm that API errors (like Tavily rate limits or missing OpenAI keys) trigger graceful, informative failures rather than hard crashes.
* **Secret Management:** Leverage 1Password CLI to dynamically inject credentials from a `.env.tpl` template into the local `.env` file.