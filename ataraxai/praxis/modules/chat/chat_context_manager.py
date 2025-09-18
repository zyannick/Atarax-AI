import uuid
from transformers import AutoTokenizer
from typing import Optional, Dict, List, Any

from ataraxai.praxis.modules.chat.chat_database_manager import (
    ChatDatabaseManager,
    Message,
)
from ataraxai.praxis.utils.vault_manager import VaultManager


class ChatContextManager:
    def __init__(self, db_manager: ChatDatabaseManager, vault_manager: VaultManager, model_name: str = "gpt2"):
        """
        Initializes the ChatContextManager with a database manager and a specified model tokenizer.

        Args:
            db_manager (ChatDatabaseManager): The database manager instance for chat data.
            model_name (str, optional): The name of the pretrained model to load the tokenizer from. Defaults to "gpt2".

        Attributes:
            db_manager (ChatDatabaseManager): Stores the provided database manager.
            tokenizer (Optional[AutoTokenizer]): The tokenizer loaded from the specified model, or None if loading fails.
            max_tokens (int): The maximum number of tokens supported by the tokenizer. Defaults to 4096 if tokenizer loads successfully, otherwise 2048.

        Raises:
            Prints a warning if the tokenizer cannot be loaded and sets tokenizer to None.
        """
        self.db_manager = db_manager
        self.vault_manager = vault_manager

        try:
            self.tokenizer: Optional[AutoTokenizer] = AutoTokenizer.from_pretrained(model_name)  # type: ignore
            self.max_tokens = 4096
        except Exception:
            print(
                "Warning: Could not load tokenizer. Context length will be estimated."
            )
            self.tokenizer = None
            self.max_tokens = 2048

    async def add_message(self, session_id: uuid.UUID, role: str, content: str):
        """
        Adds a message to the chat session.

        Args:
            session_id (uuid.UUID): The unique identifier of the chat session.
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message to be added.

        Returns:
            None
        """
        encrypted_content = self.vault_manager.encrypt(content.encode("utf-8"))
        await self.db_manager.add_message(session_id, role, encrypted_content)

    async def get_messages_for_session(self, session_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Retrieve all messages associated with a given session.

        Args:
            session_id (uuid.UUID): The unique identifier for the chat session.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a message with keys:
                - 'role': The role of the message sender (e.g., 'user', 'assistant').
                - 'content': The content of the message.
                - 'date_time': The time the message was sent.
        """
        messages: List[Message] = await self.db_manager.get_messages_for_session(session_id)
        dict_messages: List[Dict[str, Any]] = [
            {"role": msg.role, "content": self.vault_manager.decrypt(bytes(msg.content)).decode("utf-8"), "date_time": msg.date_time}
            for msg in messages
        ]
        return dict_messages

    async def get_formatted_context_for_model(
        self, session_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """
        Retrieves and formats the conversation history for a given session, preparing it for model input.

        Args:
            session_id (uuid.UUID): The unique identifier for the chat session.

        Returns:
            List[Dict[str, Any]]: A list of message dictionaries, each containing the role, content, and date_time,
            ordered chronologically and truncated to fit within the maximum token limit.

        Notes:
            - If a tokenizer is provided, it is used to count tokens; otherwise, tokens are estimated by word count.
            - The conversation history is truncated from the oldest messages if the token limit is exceeded.
        """
        messages = await self.db_manager.get_messages_for_session(session_id)

        formatted_history: List[Dict[str, Any]] = []
        current_token_count = 0

        for msg in reversed(messages):
            decrypted_content = self.vault_manager.decrypt(bytes(msg.content)).decode("utf-8")
            if not decrypted_content:
                continue
            message_dict: Dict[str, Any] = {
                "role": msg.role,
                "content": decrypted_content,
                "date_time": msg.date_time,
            }

            if self.tokenizer:
                message_token_count = len(self.tokenizer.encode(str(decrypted_content)))  # type: ignore
            else:
                message_token_count = len(str(decrypted_content).split())

            if current_token_count + message_token_count > self.max_tokens:
                print(
                    f"Context limit reached. Truncating conversation history for session {session_id}."
                )
                break

            formatted_history.append(message_dict)
            current_token_count += message_token_count

        return list(reversed(formatted_history))
