# FYP_Workbench/view_model.py
from typing import Optional, Iterator
from fyp_service import FYPService
from user_manager import UserManager, User
from data_types import CRAGResult, SourceNode, ChatMessage
from dataclasses import dataclass, field
import history_manager


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

            # CHANGED: LOAD HISTORY INSTEAD OF CLEARING
            self.chat_history = history_manager.load_history(user.username)

            # Also rebuild the string history for the LLM context
            self._string_history = [
                f"{msg.role.capitalize()}: {msg.content}"
                for msg in self.chat_history
            ]

            return True
        else:
            self.status_message = "Invalid Credentials"
            return False

    def logout(self):
        # Optional: Save one last time before logout
        if self.current_user:
            history_manager.save_history(self.current_user.username, self.chat_history)

        self.current_user = None
        self.chat_history.clear()
        self._string_history.clear()
        self.status_message = "Logged Out"

    # --- STREAMING CHAT ---
    def send_message(self, text: str) -> Iterator[ChatMessage]:
        """
        Yields the SAME ChatMessage object repeatedly, but with updated content.
        """
        if not self.current_user:
            self.status_message = "You must log in first."
            return

        # 1. User Message
        user_msg = ChatMessage(role="user", content=text)
        self.chat_history.append(user_msg)
        self._string_history.append(f"User: {text}")
        yield user_msg  # Yield user msg once so UI shows it

        # 2. Prepare AI Message (Empty)
        ai_msg = ChatMessage(role="ai", content="", confidence=0.0)
        self.chat_history.append(ai_msg)

        # 3. Call Service (Streaming)
        stream = self._service.answer(text, self.current_user, history=self._string_history)

        full_response_text = ""

        for chunk in stream:
            # Case A: Metadata (First chunk)
            if isinstance(chunk, CRAGResult):
                ai_msg.debug_sources = chunk.source_nodes
                ai_msg.confidence = chunk.confidence
                # Yield immediately to show sources/loading state
                yield ai_msg

            # Case B: Text Token
            elif isinstance(chunk, str):
                full_response_text += chunk
                ai_msg.content = full_response_text
                # Yield update to show typing effect
                yield ai_msg

        # 4. Finalize
        self._string_history.append(f"AI: {full_response_text}")
        history_manager.save_history(self.current_user.username, self.chat_history)

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