from typing import Any, Dict

from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask
from ataraxai.app_logic.modules.prompt_engine.context_manager import ContextManager, TaskContext
from ataraxai.app_logic.modules.prompt_engine.prompt_manager import PromptManager
from ataraxai import core_ai_py  # type: ignore
from ataraxai.app_logic.modules.chat.chat_database_manager import ChatDatabaseManager
from ataraxai.app_logic.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager

class StandardChatTask(BaseTask):
    def __init__(self):
        self.id = "standard_chat"
        self.description = "Performs a standard chat interaction with RAG context."
        self.required_inputs = ["user_query", "session_id"]
        self.prompt_template_name = "main_chat"
        super().__init__()

    def _load_resources(self) -> None:
        pass  

    def execute(
        self,
        processed_input: Dict[str, Any],
        context: TaskContext,  
        prompt_manager: PromptManager,
        core_ai_service: core_ai_py.CoreAIService, # type: ignore
        generation_params: Dict[str, Any],
        chat_db_manager: ChatDatabaseManager,
        rag_manager: AtaraxAIRAGManager,
    ) -> str:

        user_query = processed_input["user_query"]
        session_id = processed_input["session_id"]

        chat_db_manager.add_message(session_id, role="user", content=user_query)

        conversation_history = chat_db_manager.get_messages_for_session(session_id)
        rag_results = rag_manager.query_knowledge(query_text=user_query, n_results=3)

        rag_context = (
            "\n".join(rag_results["documents"][0])
            if rag_results and rag_results["documents"]
            else "No relevant documents found."
        )

        final_prompt = prompt_manager.load_template(
            self.prompt_template_name, # type: ignore
            history=conversation_history,
            context=rag_context,
            query=user_query,
        )

        model_response_text : str = core_ai_service.process_prompt(  # type: ignore
                final_prompt, generation_params
            )

        chat_db_manager.add_message(
            session_id, role="assistant", content=model_response_text
        )

        return model_response_text # type: ignore
