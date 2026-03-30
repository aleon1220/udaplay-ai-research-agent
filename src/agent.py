import logging
from typing import List, Dict, Any, Sequence, Annotated
from pydantic import BaseModel, Field, ConfigDict
import operator
from typing import Annotated, Sequence, TypedDict, Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


from src.tools import retrieve_game, evaluate_retrieval, game_web_search, persist_web_search_results

logger = logging.getLogger(__name__)

# --- Define the State ---
class AgentState(TypedDict):
    chat_history: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    context: str
    retrieval_sufficient: bool
    final_answer: str
    structured_response: Dict[str, Any]

# --- Define the Output Schema ---
# now OpenAI is more strict some tweakings happening here.
class Fact(BaseModel):
    # This explicitly tells OpenAI that no other fields are allowed
    model_config = ConfigDict(extra="forbid") 
    
    attribute: str = Field(description="The specific attribute or entity being extracted.")
    value: str = Field(description="The extracted value for that attribute.")

class FinalAnswerResponse(BaseModel):
    # Lock down the parent model as well
    model_config = ConfigDict(extra="forbid") 
    
    natural_language_answer: str = Field(description="Answer in clear natural language, includes citations.")
    structured_data: List[Fact] = Field(description="A comprehensive list of extracted facts from the context.")

# --- Define Nodes ---
def retrieve_node(state: AgentState):
    """Node that attempts to answer using internal knowledge, contextualizing the question if history exists."""
    logger.info("Executing Node: retrieve_node")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    # If there is history, we rewrite the question to be standalone for better retrieval
    if chat_history:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history[-5:]]) # Use last 5 messages
        
        rewrite_prompt = f"""
        Given the following chat history and a follow-up question, rephrase the follow-up question to be a standalone question that can be used to search a database.
        
        Chat History:
        {history_str}
        
        Follow-up Question: {question}
        
        Standalone Question:"""
        
        try:
            standalone_question = llm.invoke(rewrite_prompt).content.strip()
            logger.info(f"Rewritten question for retrieval: '{standalone_question}'")
            search_query = standalone_question
        except Exception as e:
            logger.error(f"Failed to rewrite question: {e}")
            search_query = question
    else:
        search_query = question

    context = retrieve_game(search_query)
    return {"context": context}

def evaluate_node(state: AgentState):
    """Node that evaluates if the retrieved internal data is sufficient."""
    logger.info("Executing Node: evaluate_node")
    question = state["question"]
    context = state["context"]
    evaluation = evaluate_retrieval(question, context)
    return {"retrieval_sufficient": evaluation.has_sufficient_info}

def web_search_node(state: AgentState):
    """Node that falls back to web search if internal data is insufficient."""
    logger.info("Executing Node: web_search_node")
    question = state["question"]
    context = state["context"]
    
    web_results = game_web_search(question)
    
    if web_results:
        formatted_web_results = "\n\n---\n\n".join(
            [f"Source: {res.get('title')} ({res.get('url')})\nContent: {res.get('content')}" for res in web_results]
        )
        updated_context = f"{context}\n\n=== WEB SEARCH RESULTS ===\n{formatted_web_results}"
        
        # Bonus: persist the newly found information into long-term memory
        persist_web_search_results(web_results, question)
    else:
        updated_context = f"{context}\n\n=== WEB SEARCH RESULTS ===\nNo relevant web results found."
        
    return {"context": updated_context}

def generate_answer_node(state: AgentState):
    """Node that generates the final natural language answer and structured JSON."""
    logger.info("Executing Node: generate_answer_node")
    question = state["question"]
    context = state["context"]
    chat_history = state.get("chat_history", [])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    structured_llm = llm.with_structured_output(FinalAnswerResponse)
    
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history]) if chat_history else "No previous history."
    
    # previously doing extraction to a JSON file
    prompt = f"""
    You are UdaPlay, an expert AI Research Agent for a gaming analytics company.
    Answer the user's question based on the provided context.
    If the context contains web search results, be sure to cite the URLs.
    If the context contains internal database results, cite the internal IDs.
    
    Chat History:
    {history_str}
    
    Context:
    {context}
    
    User Question: {question}
    
    Provide the answer in natural language. THEN, extract the key facts into the structured_data list.
    Each item in structured_data must be a specific fact with an 'attribute' name and its corresponding 'value'.
    """
    
    response = structured_llm.invoke(prompt)
    logger.info("Final answer generated successfully.")
    
    return {
        "final_answer": response.natural_language_answer,
        "structured_response": response.structured_data,
        "chat_history": [HumanMessage(content=question), AIMessage(content=response.natural_language_answer)]
    }

# --- Define Edges ---
def should_web_search(state: AgentState):
    """Conditional edge logic to determine if a web search is needed."""
    if state["retrieval_sufficient"]:
        logger.info("Edge Evaluation: Information is sufficient. Proceeding to generate_answer.")
        return "generate_answer"
    else:
        logger.info("Edge Evaluation: Information is insufficient. Proceeding to web_search.")
        return "web_search"

# --- Build the Graph ---
def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # Define execution flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "evaluate")
    
    # Conditional logic
    workflow.add_conditional_edges(
        "evaluate",
        should_web_search,
        {
            "generate_answer": "generate_answer",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile(checkpointer=checkpointer)

# --- Agent Interface ---
class UdaPlayAgent:
    """
    Stateful AI Research Agent that maintains conversation context across multiple queries using session IDs.
    """
    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = build_graph(checkpointer=self.checkpointer)
        
    def invoke(self, question: str, session_id: str = "default_session") -> Dict[str, Any]:
        """
        Invokes the agent workflow with a user question and session ID.
        
        Args:
            question (str): The user's query.
            session_id (str): The unique identifier for the conversation session.
            
        Returns:
            Dict[str, Any]: The final state of the workflow containing the answer and structured response.
        """
        logger.info(f"\n--- Processing Query: '{question}' (Session: {session_id}) ---")
        
        config = {"configurable": {"thread_id": session_id}}
        
        initial_state = {
            "question": question,
            "context": "",
            "retrieval_sufficient": False,
            "final_answer": "",
            "structured_response": {}
        }
        
        result = self.graph.invoke(initial_state, config=config)
        
        return result

