import datetime
from typing_extensions import Optional, Dict, Any
from ataraxai.praxis.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager
from typing import Callable, List


class TaskContext:

    def __init__(
        self,
        user_query: str,
        user_history: Optional[list[str]] = None,
        task_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the context manager with the user's query, history, and task information.

        Args:
            user_query (str): The current query from the user.
            user_history (Optional[list[str]], optional): A list of previous user queries or interactions. Defaults to an empty list if not provided.
            task_info (Optional[Dict[str, Any]], optional): Additional information related to the current task. Defaults to an empty dictionary if not provided.
        """
        self.user_query = user_query
        self.user_history = user_history if user_history is not None else []
        self.task_info = task_info if task_info is not None else {}

    def add_to_history(self, entry: str):
        """
        Add a new entry to the user's history.

        Args:
            entry (str): The entry to be added to the user history.
        """
        self.user_history.append(entry)

    def get_context_summary(self) -> str:
        """
        Generates a summary string containing the user's query, conversation history, and task information.

        Returns:
            str: A formatted string summarizing the current context, including the user query, history, and task information.
        """
        return f"Query: {self.user_query}, History: {self.user_history}, Task Info: {self.task_info}"


class ContextManager:
    def __init__(self, config: Dict[str, Any], rag_manager: AtaraxAIRAGManager):
        """
        Initializes the context manager with the provided configuration and RAG (Retrieval-Augmented Generation) manager.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the context manager.
            rag_manager (AtaraxAIRAGManager): Instance of the RAG manager to handle retrieval-augmented tasks.

        Attributes:
            config (Dict[str, Any]): Stores the configuration settings.
            rag_manager (AtaraxAIRAGManager): Reference to the RAG manager instance.
            context_providers (Dict[str, Callable]): Maps context provider names to their corresponding methods for retrieving context data.
        """

        self.config = config
        self.rag_manager = rag_manager

        self.context_providers: Dict[str, Callable] = {
            "current_date": self._get_current_date,
            "default_role_prompt": self._get_default_role_prompt,
            "relevant_document_chunks": self._get_relevant_document_chunks,
            "user_calendar_today": self._get_calendar_events_today,
            "file_content": self._get_file_content,
        }

    def get_context(
        self,
        context_key: str,
        user_inputs: Optional[Dict] = None,
    ) -> Any:
        if context_key == "current_date":
            return self._get_current_date()
        elif context_key == "default_role_prompt":
            current_role = self.config.get("current_user_role", "default_user")
            persona_key = (
                self.config.get("roles", {})
                .get(current_role, {})
                .get("default_persona_prompt_key")
            )
            return self.config.get("personas", {}).get(
                persona_key, "You are a helpful AI assistant."
            )
        elif (
            context_key == "relevant_document_chunks"
            and user_inputs
            and "query" in user_inputs
        ):
            print(user_inputs["query"])
            return self._get_relevant_document_chunks(user_inputs["query"])
        elif context_key == "user_calendar_today":
            return self._get_calendar_events_today()
        elif (
            context_key == "file_content" and user_inputs and "file_path" in user_inputs
        ):
            return self._get_file_content(user_inputs["file_path"])
        else:
            return None

    def _get_relevant_document_chunks(self, user_inputs: Dict[str, Any]) -> List[str]:
        query: Optional[str] = user_inputs.get("query", None)
        if not query:
            return []

        query_results: List[str] = self.rag_manager.query_knowledge(query_text=query)

        return query_results if query_results else []

    def _get_current_date(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _get_calendar_events_today(self) -> list[str]:
        # TODO: Implement the logic to retrieve calendar events for today.
        return []

    def _get_file_content(self, file_path: str) -> str | None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def _get_default_role_prompt(self) -> str:
        current_role = self.config.get("current_user_role", "default_user")
        persona_key = (
            self.config.get("roles", {})
            .get(current_role, {})
            .get("default_persona_prompt_key")
        )
        return self.config.get("personas", {}).get(
            persona_key, "You are a helpful AI assistant."
        )
