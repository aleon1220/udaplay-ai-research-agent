import os
import logging
import hashlib
from typing import Dict, Any, List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from src.database import GameDatabase

logger = logging.getLogger(__name__)

# Initialize the database instance globally so tools can use it
db = GameDatabase()

def retrieve_game(query: str, n_results: int = 3) -> str:
    """
    Queries the internal local ChromaDB vector store for game information.
    """
    logger.info(f"Retrieving games for query: '{query}'")
    results = db.search(query, n_results=n_results)
    
    if not results:
        return "No relevant game information found in the internal database."
        
    formatted_results = []
    for res in results:
        formatted_results.append(f"[{res.get('id', 'N/A')}] {res['text']}")
        
    return "\n\n---\n\n".join(formatted_results)

class RetrievalEvaluation(BaseModel):
    has_sufficient_info: bool = Field(description="True if the retrieved context contains enough information to fully answer the user's question, False otherwise.")
    reasoning: str = Field(description="Explanation of why the context is or is not sufficient.")

def evaluate_retrieval(question: str, context: str) -> RetrievalEvaluation:
    """
    Assesses if the retrieved internal data answers the user's question with high confidence.
    """
    logger.info("Evaluating retrieval sufficiency...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(RetrievalEvaluation)
    
    prompt = f"""
    You are an evaluator assessing if the retrieved context is sufficient to answer the user's question.
    
    User Question: {question}
    
    Retrieved Context:
    {context}
    
    Determine if the context contains enough information to fully and accurately answer the question.
    """
    
    result = structured_llm.invoke(prompt)
    logger.info(f"Evaluation result: Sufficient={result.has_sufficient_info}, Reasoning: {result.reasoning}")
    return result

def game_web_search(query: str) -> List[Dict[str, str]]:
    """
    Uses the Tavily API to search the web for game information.
    """
    logger.info(f"Web search for query: '{query}'")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("TAVILY_API_KEY environment variable not set. Cannot perform web search.")
        return []
        
    try:
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(query=query, search_depth="advanced", max_results=3)
        
        results = response.get("results", [])
        return results
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []

def persist_web_search_results(results: List[Dict[str, str]], query: str):
    """
    Persists the newly found information into long-term memory.
    """
    if not results:
        return
        
    docs_to_insert = []
    for i, res in enumerate(results):
        url = res.get('url', str(i))
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        
        doc = {
            "id": f"web_{url_hash}",
            "title": res.get("title", f"Web Result for {query}"),
            "description": res.get("content", ""),
            "url": url,
            "source": "web_search",
            "platforms": "Unknown",  # Provide defaults for DB compatibility
            "publisher": "Unknown",
            "release_date": "Unknown"
        }
        docs_to_insert.append(doc)
        
    logger.info(f"Persisting {len(docs_to_insert)} web search results to ChromaDB.")
    db.insert_documents(docs_to_insert)
