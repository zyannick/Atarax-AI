import datetime
from typing_extensions import Optional, Dict, Any
from ataraxai.app_logic.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager
from typing import Callable, List


class TaskContext:

    def __init__(
        self,
        user_query: str,
        user_history: Optional[list[str]] = None,
        task_info: Optional[Dict[str, Any]] = None,
    ):
        self.user_query = user_query
        self.user_history = user_history if user_history is not None else []
        self.task_info = task_info if task_info is not None else {}

    def add_to_history(self, entry: str):
        self.user_history.append(entry)

    def get_context_summary(self) -> str:
        return f"Query: {self.user_query}, History: {self.user_history}, Task Info: {self.task_info}"


class ContextManager:
    def __init__(self, config: Dict[str, Any], rag_manager: AtaraxAIRAGManager):

        self.config = config
        self.rag_manager = rag_manager

        self.context_providers: Dict[str, Callable] = {
            "current_date": self._get_current_date,
            "default_role_prompt": self._get_default_role_prompt,
            "relevant_document_chunks": self._get_relevant_document_chunks,
            "user_calendar_today": self._get_calendar_events_today,
            "file_content": self._get_file_content,
        }

    def _get_relevant_document_chunks(self, user_inputs: Dict[str, Any]) -> List[str]:  # type: ignore
        query: Optional[str] = user_inputs.get("query", None)
        if not query:
            return []

        query_results: List[str] = self.rag_manager.query_knowledge(
            query_text=query
        )

        if (
            query_results
            and "documents" in query_results
            and query_results["documents"] # type: ignore
        ):
            return query_results["documents"][0] # type: ignore

        return []

    def get_context(
        self,
        context_key: str,
        user_inputs: Optional[Dict] = None,
        task_info: Optional[Dict] = None,
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
