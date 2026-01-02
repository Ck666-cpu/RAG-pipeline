# FYP_Workbench/my_brain.py
from data_type import CRAGResult
import time  # Just for effect


class MyFYPBot:
    def __init__(self):
        # INITIALIZATION: Load your heavy AI models, API keys, or databases here.
        print(" [Bot] Loading FYP Intelligence Module...")
        # self.model = load_my_model()
        print(" [Bot] Ready!")

    def answer(self, question: str, user_role: str) -> CRAGResult:
        print(f" [Bot] Thinking about: '{question}' for user: {user_role}...")

        # --- PLACE YOUR REAL LOGIC HERE ---
        # 1. Search your database
        # 2. Call your LLM
        # 3. Generate text

        # For now, I will pretend to be your bot:
        generated_answer = f"This is a test answer. I heard you ask about '{question}'."

        # --- END OF REAL LOGIC ---

        return CRAGResult(
            answer=generated_answer,
            sources=["my_fyp_database.pdf", "test_data.txt"],
            confidence=0.95
        )