import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 3. Import and initialize
# Load environment variables
load_dotenv()

class GameDatabase:
    """
    Handles loading JSON game data, setting up a persistent ChromaDB,
    embedding data, and providing semantic search capabilities.
    """
    
    def __init__(self, db_dir: str = "vector_db", collection_name: str = "games"):
        """
        Initializes the GameDatabase.
        
        Args:
            db_dir (str): Directory to store the persistent ChromaDB.
            collection_name (str): Name of the ChromaDB collection.
        """
        self.db_dir = Path(db_dir)
        self.collection_name = collection_name
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        
        # Initialize OpenAI embedding function
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment. Using default embeddings if fallback needed.")
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )
            
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
        logger.info(f"Initialized ChromaDB collection: '{self.collection_name}' at '{self.db_dir}'")

    def load_json_files(self, data_dir: str = "data/raw") -> List[Dict[str, Any]]:
        """
        Loads and parses all JSON files from the specified directory.
        
        Args:
            data_dir (str): Directory containing raw JSON files.
            
        Returns:
            List[Dict[str, Any]]: List of parsed JSON documents (dictionaries).
        """
        data_path = Path(data_dir)
        documents = []
        
        if not data_path.exists():
            logger.warning(f"Data directory '{data_dir}' does not exist.")
            return documents
            
        for file_path in data_path.rglob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        documents.extend(data)
                    elif isinstance(data, dict):
                        documents.append(data)
                logger.info(f"Loaded {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return documents

    def format_document(self, doc: Dict[str, Any]) -> str:
        """
        Formats a JSON document into a text string suitable for embedding.
        
        Args:
            doc (Dict[str, Any]): The raw game dictionary.
            
        Returns:
            str: Formatted text representation of the game.
        """
        # Basic formatting, assuming standard fields like title, release_date, etc.
        # This can be adjusted based on the actual JSON structure.
        title = doc.get("title", doc.get("Name", doc.get("name", "Unknown Title")))
        desc = doc.get("description", doc.get("Description", doc.get("summary", "")))
        release_date = doc.get("release_date", doc.get("YearOfRelease", "Unknown Date"))
        platforms = doc.get("platforms", doc.get("Platform", []))
        if isinstance(platforms, list):
            platforms_str = ", ".join(platforms)
        else:
            platforms_str = str(platforms)
            
        publisher = doc.get("publisher", doc.get("Publisher", "Unknown Publisher"))
        
        formatted_text = f"Title: {title}\nRelease Date: {release_date}\nPlatforms: {platforms_str}\nPublisher: {publisher}\nDescription: {desc}"
        return formatted_text

    def insert_documents(self, documents: List[Dict[str, Any]]):
        """
        Embeds and inserts formatted documents into ChromaDB.
        
        Args:
            documents (List[Dict[str, Any]]): List of raw game dictionaries.
        """
        if not documents:
            logger.warning("No documents to insert.")
            return

        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            # Try to get a unique ID, otherwise generate one
            doc_id = str(doc.get("id", f"game_{i}"))
            formatted_text = self.format_document(doc)
            
            # Clean metadata (ChromaDB requires metadata values to be str, int, float, or bool)
            metadata = {}
            for k, v in doc.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
                elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                     metadata[k] = ", ".join(v)
                     
            ids.append(doc_id)
            texts.append(formatted_text)
            metadatas.append(metadata)

        # Batch insert to handle API limits
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            self.collection.upsert(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logger.info(f"Inserted batch {i//batch_size + 1} into ChromaDB.")

    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Performs semantic search on the vector database.
        
        Args:
            query (str): The search query.
            n_results (int): Number of results to return.
            
        Returns:
            List[Dict[str, Any]]: A list of search results including text and metadata.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "id": results['ids'][0][i] if results['ids'] else None,
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results.get('distances') else None
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

if __name__ == "__main__":
    # Example usage / Verification
    db = GameDatabase()
    
    raw_data = db.load_json_files()
    if raw_data:
        db.insert_documents(raw_data)
        
    print("\nTesting Search:")
    search_results = db.search("What is the release date of the latest action game?", n_results=2)
    for res in search_results:
        print(f"ID: {res['id']}\nText: {res['text']}\nDistance: {res['distance']}\n---")
