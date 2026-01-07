# FYP_Workbench/view_model.py
from typing import Optional
from fyp_service import FYPService
from user_manager import UserManager, User
from data_types import CRAGResult, SourceNode
from dataclasses import dataclass, field

@dataclass
class ChatMessage:
    role: str
    content: str
    debug_sources: list = None
    confidence: float = 0.0


class ChatViewModel:
    def __init__(self):
        self._service = FYPService()
        self._user_manager = UserManager()

        # SESSION STATE
        self.current_user: Optional[User] = None

        self.chat_history: list[ChatMessage] = []
        self.status_message: str = "Please Log In"

    # --- AUTHENTICATION ---
    def login(self, username, password) -> bool:
        user = self._user_manager.login(username, password)
        if user:
            self.current_user = user
            self.status_message = f"Welcome, {user.role} {user.username}"
            self.chat_history.clear()  # Clear view on relogin
            return True
        else:
            self.status_message = "Invalid Credentials"
            return False

    def logout(self):
        self.current_user = None
        self.chat_history.clear()
        self.status_message = "Logged Out"

    # --- CHAT (Permission Checked) ---
    def send_message(self, text: str):
        if not self.current_user:
            self.status_message = "You must log in first."
            return None

        # Delegate to Service (which handles logic)
        # We pass the WHOLE user object now
        result = self._service.answer(text, self.current_user)

        ai_msg = ChatMessage(
            role="ai",
            content=result.answer,
            debug_sources=result.source_nodes,
            confidence=result.confidence
        )
        self.chat_history.append(ai_msg)
        return ai_msg

    # --- USER MANAGEMENT (Master Admin/Admin Features) ---
    def register_user(self, new_user, new_pass, role):
        if not self.current_user: return "Not Logged In"
        try:
            return self._user_manager.register_user(
                self.current_user.role, new_user, new_pass, role
            )
        except Exception as e:
            return str(e)

    def update_role(self, target_user, new_role):
        if not self.current_user: return "Not Logged In"
        try:
            return self._user_manager.update_role(
                self.current_user.role, target_user, new_role
            )
        except Exception as e:
            return str(e)

    def delete_user(self, target_user):
        if not self.current_user: return "Not Logged In"
        try:
            return self._user_manager.delete_user(
                self.current_user.role, target_user
            )
        except Exception as e:
            return str(e)

    # --- DOCUMENT UPLOAD ---
    def upload_document(self, file_path, is_global=False):
        if not self.current_user: return "Not Logged In"

        success, msg = self._service.upload_document(
            file_path, self.current_user, is_global
        )
        self.status_message = msg
        return msg