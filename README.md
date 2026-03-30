# UdaPlay AI Research Agent

UdaPlay is a stateful AI Research Agent designed for a gaming analytics company. It answers natural language questions about video games using a robust two-tier retrieval system (Local RAG + Web Search Fallback) and maintains conversation state across multiple queries.

## Features
- **Local RAG Pipeline:** Ingests game data into a persistent ChromaDB vector store.
- **Evaluated Web Fallback:** Uses an LLM as a judge to determine if internal database answers are sufficient; if not, it falls back to the Tavily Web Search API.
- **Long-term Memory:** Automatically persists newly discovered web information into the local vector database for future queries.
- **Stateful Conversations:** Built with LangGraph to maintain chat history and context across user sessions.
- **Structured Output:** Returns answers in both readable natural language and structured JSON formats.

## Prerequisites & Setup
### Create a Virtual Environment
1. set up the virtual env with python module
- go to directory
```powershell
cd notebooks
```

- enable the virtual enfv
```powershell
python -m venv udacity_udaplay_agent
```

2. Activate environment
```powershell
.\udacity_udaplay_agent\Scripts\activate.bat
```

3. Set up the app dependencies
```powershell
python -m pip install -r requirements.txt
```

4. inject secrets to a target directory
```powershell
op inject -i .env.tpl -o .env
```

### Check dependencies
1. **Install Dependencies:**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   check the `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## Running the Project

The project workflow is demonstrated step-by-step using Jupyter notebooks located in the `notebooks/` directory.

### Step 1: Data Ingestion (RAG Setup)
To initialize the vector database and load the raw game data:
1. Open `notebooks/Udaplay_01_solution_project.ipynb` in your preferred notebook environment.
2. Run all cells. This script will recursively parse the `data/raw/` directory, map properties to the internal structure, generate embeddings, and store them in the persistent `vector_db/` folder.

### Step 2: Agent Execution & Testing
To interact with the UdaPlay agent and test its semantic retrieval, web fallback, and conversational memory capabilities:
1. Open `notebooks/Udaplay_02_solution_project.ipynb`.
2. Run all cells. The execution logs will print the internal "thought process" (e.g., retrieving context, deciding if it's sufficient, triggering web searches) and output the final, structured answers.

## License
[License](LICENSE)
