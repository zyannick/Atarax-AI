import uuid
from transformers import AutoTokenizer
from typing import Optional, Dict, List, Any

from ataraxai.app_logic.modules.chat.chat_database_manager import (
    ChatDatabaseManager,
    Message,
)


class ChatContextManager:
    def __init__(self, db_manager: ChatDatabaseManager, model_name: str = "gpt2"):
        self.db_manager = db_manager

        try:
            self.tokenizer: Optional[AutoTokenizer] = AutoTokenizer.from_pretrained(model_name)  # type: ignore
            self.max_tokens = 4096
        except Exception:
            print(
                "Warning: Could not load tokenizer. Context length will be estimated."
            )
            self.tokenizer = None
            self.max_tokens = 2048

    def add_message(self, session_id: uuid.UUID, role: str, content: str):
        self.db_manager.add_message(session_id, role, content)

    def get_messages_for_session(self, session_id: uuid.UUID) -> List[Dict[str, Any]]:
        messages: List[Message] = self.db_manager.get_messages_for_session(session_id)
        dict_messages: List[Dict[str, Any]] = [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
            for msg in messages
        ]
        return dict_messages

    def get_formatted_context_for_model(
        self, session_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        messages = self.db_manager.get_messages_for_session(session_id)

        formatted_history: List[Dict[str, Any]] = []
        current_token_count = 0

        for msg in reversed(messages):
            message_dict: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
            }

            if self.tokenizer:
                message_token_count = len(self.tokenizer.encode(str(msg.content)))  # type: ignore
            else:
                message_token_count = len(str(msg.content).split())

            if current_token_count + message_token_count > self.max_tokens:
                print(
                    f"Context limit reached. Truncating conversation history for session {session_id}."
                )
                break

            formatted_history.append(message_dict)
            current_token_count += message_token_count

        return list(reversed(formatted_history))
