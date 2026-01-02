# FYP_Workbench/my_brain.py
import ollama
from data_type import CRAGResult
import time  # Just for effect


class MyFYPBot:
    def __init__(self):
        print(" [Bot] Loading FYP Intelligence Module...")
        self.model_name = "phi3"
        print(f" [Bot] Ready! :{self.model_name}")

    def answer(self, question: str, user_role: str) -> CRAGResult:
        print(f" [Bot] Thinking about: '{question}' for user: {user_role}...")


        # construct a prompt
        chat_prompt = (
            f"You are a helpful Real Estate assistant. "
            f"The user interacting with you has the role: '{user_role}'."
            "Keep your answers concise and professional. "
        )

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': chat_prompt},
                {'role': user_role, 'content': question},
            ])

            return_answer = response["messages"]["content"]

            return CRAGResult(
                answer=return_answer,
                sources=[],
                confidence=1.0 #adjust after fine-tune
            )
        except Exception as e:
            print(f"[Error] Failed to talk to Ollama {e}")
            return CRAGResult(
                answer="Sorry, It seem to be something wrong",
                sources=[],
                confidence=0.0
            )



