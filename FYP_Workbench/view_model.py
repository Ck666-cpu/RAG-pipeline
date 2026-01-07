# FYP_Workbench/view_model.py
from dataclasses import dataclass, field
from typing import List, Optional
from fyp_service import FYPService
from data_types import CRAGResult


@dataclass
class ChatMessage:
    role: str
    content: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0


class ChatViewModel:
    def __init__(self):
        self._service = FYPService()

        # STATE
        self.chat_history: List[ChatMessage] = []
        self.is_processing: bool = False
        self.status_message: str = "Ready"

        # NEW: Role Management
        self.current_role: str = "User"  # Default role

        self._string_history: List[str] = []

    def set_user_role(self, role: str):
        """Called by UI to switch roles (e.g., 'Admin' or 'Tenant')"""
        self.current_role = role
        self.status_message = f"Role switched to {role}"
        # Optional: Clear history on role switch to avoid context confusion
        # self.clear_history()

    def send_message(self, user_input: str) -> Optional[ChatMessage]:
        if not user_input.strip():
            return None

        self.is_processing = True
        self.status_message = f"Thinking ({self.current_role})..."

        user_msg = ChatMessage(role="user", content=user_input)
        self.chat_history.append(user_msg)
        self._string_history.append(f"User: {user_input}")

        try:
            # CALL SERVICE WITH CURRENT ROLE
            result: CRAGResult = self._service.answer(
                question=user_input,
                user_role=self.current_role,  # <--- PASSING THE STATE
                history=self._string_history
            )

            ai_msg = ChatMessage(
                role="ai",
                content=result.answer,
                sources=result.sources,
                confidence=result.confidence
            )

            self.chat_history.append(ai_msg)
            self._string_history.append(f"AI: {result.answer}")
            self.status_message = "Ready"

            self.is_processing = False
            return ai_msg

        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.is_processing = False
            return ChatMessage(role="ai", content="An internal error occurred.")

    def upload_document(self, file_path: str) -> str:
        self.is_processing = True
        self.status_message = "Uploading..."
        try:
            success, message = self._service.upload_document(file_path)
            self.status_message = message
        except Exception as e:
            self.status_message = f"Upload Error: {str(e)}"

        self.is_processing = False
        return self.status_message

    def clear_history(self):
        self.chat_history.clear()
        self._string_history.clear()
        self.status_message = "Memory cleared."