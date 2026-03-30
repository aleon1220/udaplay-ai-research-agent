import logging
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.agent import UdaPlayAgent
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load env from notebooks/.env
load_dotenv('notebooks/.env')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_sessions():
    agent = UdaPlayAgent()
    
    print("\n--- Testing Session 1 ---")
    session_1 = "user_123"
    q1 = "What platforms is 'Cyberpunk 2077' available on?"
    print(f"Session {session_1}, Query: {q1}")
    res1 = agent.invoke(q1, session_id=session_1)
    print(f"Answer: {res1['final_answer']}")
    
    q2 = "What was its original release date?"
    print(f"\nSession {session_1}, Query: {q2} (Should remember Cyberpunk 2077)")
    res2 = agent.invoke(q2, session_id=session_1)
    print(f"Answer: {res2['final_answer']}")
    
    print("\n--- Testing Session 2 (Independent) ---")
    session_2 = "user_456"
    q3 = "Who developed 'Elden Ring'?"
    print(f"Session {session_2}, Query: {q3}")
    res3 = agent.invoke(q3, session_id=session_2)
    print(f"Answer: {res3['final_answer']}")
    
    q4 = "What was its release date?"
    print(f"\nSession {session_2}, Query: {q4} (Should remember Elden Ring, NOT Cyberpunk)")
    res4 = agent.invoke(q4, session_id=session_2)
    print(f"Answer: {res4['final_answer']}")
    
    # Verify session 1 still works
    print("\n--- Verifying Session 1 still has its context ---")
    q5 = "Give me one more fact about that first game I asked about."
    print(f"Session {session_1}, Query: {q5}")
    res5 = agent.invoke(q5, session_id=session_1)
    print(f"Answer: {res5['final_answer']}")

if __name__ == "__main__":
    test_sessions()
