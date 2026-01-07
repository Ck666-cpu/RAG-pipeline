# FYP_Workbench/view_model.py
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from fyp_service import FYPService
from data_types import CRAGResult
import model_db


@dataclass
class ChatMessage:
    """Represents a single message in the UI"""
    role: str  # 'user' or 'ai'
    content: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0


class ChatViewModel:
    def __init__(self):
        # Initialize the Model (Service)
        self._service = FYPService()

        # State (The View binds to these)
        self.chat_history: List[ChatMessage] = []
        self.is_processing: bool = False
        self.status_message: str = "Ready"

        # Internal history string list for the LLM context
        self._string_history: List[str] = []

    def send_message(self, user_input: str) -> Optional[ChatMessage]:
        """
        Command to handle user input.
        Returns the AI response message to be displayed.
        """
        if not user_input.strip():
            return None

        self.is_processing = True
        self.status_message = "Thinking..."

        # 1. Add User Message to State
        user_msg = ChatMessage(role="user", content=user_input)
        self.chat_history.append(user_msg)
        self._string_history.append(f"User: {user_input}")

        try:
            # 2. Call the Model (FYPService)
            # We pass the string history so the service can 'contextualize' the query
            result: CRAGResult = self._service.answer(
                question=user_input,
                user_role="user",
                history=self._string_history
            )

            # 3. Create AI Message from Result
            ai_msg = ChatMessage(
                role="ai",
                content=result.answer,
                sources=result.sources,
                confidence=result.confidence
            )

            # 4. Update State
            self.chat_history.append(ai_msg)
            self._string_history.append(f"AI: {result.answer}")
            self.status_message = f"Answered with {result.confidence:.2f} confidence."

            self.is_processing = False
            return ai_msg

        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.is_processing = False
            # Return an error message to the UI
            return ChatMessage(role="ai", content="An internal error occurred.")

    def upload_document(self, file_path: str) -> str:
        """
        Command to upload a file. Returns a success/fail string message.
        """
        self.is_processing = True
        self.status_message = "Uploading..."

        try:
            success, message = self._service.upload_document(file_path)
            self.status_message = message
            self.is_processing = False
            return message
        except Exception as e:
            self.is_processing = False
            return f"Upload Failed: {str(e)}"

    def clear_history(self):
        """Resets the chat memory."""
        self.chat_history.clear()
        self._string_history.clear()
        self.status_message = "Memory cleared."