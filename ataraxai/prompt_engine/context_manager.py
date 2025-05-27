
import datetime


class TaskContext:

    def __init__(self, user_query: str, user_history: list[str] = None, task_info: dict = None):
        self.user_query = user_query
        self.user_history = user_history if user_history is not None else []
        self.task_info = task_info if task_info is not None else {}

    def add_to_history(self, entry: str):
        self.user_history.append(entry)

    def get_context_summary(self) -> str:
        return f"Query: {self.user_query}, History: {self.user_history}, Task Info: {self.task_info}"

class ContextManager:
    def __init__(self, db_path, faiss_index_path, config):

        self.config = config

    def get_context(self, context_key: str, user_inputs: dict = None, task_info: dict = None) -> any:
        if context_key == "current_date":
            return self._get_current_date()
        elif context_key == "default_role_prompt":
            current_role = self.config.get("current_user_role", "default_user") 
            persona_key = self.config.get("roles", {}).get(current_role, {}).get("default_persona_prompt_key")
            return self.config.get("personas", {}).get(persona_key, "You are a helpful AI assistant.")
        elif context_key == "relevant_document_chunks" and user_inputs and "query" in user_inputs:
            return self._get_relevant_document_chunks(user_inputs["query"])
        elif context_key == "user_calendar_today":
            return self._get_calendar_events_today()
        elif context_key == "file_content" and user_inputs and "file_path" in user_inputs:
            return self._get_file_content(user_inputs["file_path"])
        else:
            return None

    def _get_current_date(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _get_relevant_document_chunks(self, query: str, top_k: int = 3) -> list[str]:
        pass

    def _get_calendar_events_today(self) -> list[str]:
        pass

    def _get_file_content(self, file_path: str) -> str | None:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

