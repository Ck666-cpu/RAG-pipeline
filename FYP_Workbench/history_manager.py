# FYP_Workbench/history_manager.py
import json
import os
from dataclasses import asdict
from typing import List
from data_types import ChatMessage

HISTORY_DIR = "chat_histories"


def ensure_history_dir():
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)


def save_history(username: str, messages: List[ChatMessage]):
    """Saves the list of ChatMessage objects to a JSON file."""
    ensure_history_dir()
    file_path = os.path.join(HISTORY_DIR, f"{username}.json")

    # Convert ChatMessage objects to simple dictionaries
    data = []
    for msg in messages:
        # We use asdict to convert dataclass to dict
        # We might need to handle 'debug_sources' carefully if it's complex
        msg_dict = {
            "role": msg.role,
            "content": msg.content,
            "confidence": msg.confidence,
            # We skip saving full source nodes to keep JSON small,
            # or we can serialize them if needed.
            "sources_summary": [s.file_name for s in msg.debug_sources] if msg.debug_sources else []
        }
        data.append(msg_dict)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_history(username: str) -> List[ChatMessage]:
    """Loads chat history for a specific user."""
    ensure_history_dir()
    file_path = os.path.join(HISTORY_DIR, f"{username}.json")

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        messages = []
        for item in data:
            # Reconstruct ChatMessage objects
            # Note: We aren't reloading the full 'debug_sources' object list
            # to keep it simple, but we could if we serialized it fully.
            msg = ChatMessage(
                role=item["role"],
                content=item["content"],
                confidence=item.get("confidence", 0.0)
            )
            messages.append(msg)
        return messages
    except Exception as e:
        print(f"Error loading history: {e}")
        return []