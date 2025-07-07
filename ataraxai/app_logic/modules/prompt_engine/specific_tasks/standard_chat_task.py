from typing import Any, Dict

from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask
from ataraxai.app_logic.modules.prompt_engine.context_manager import TaskContext
from ataraxai.app_logic.modules.chat.chat_context_manager import ChatContextManager

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
        dependencies: Dict[str, Any],
    ) -> str:

        user_query = processed_input["user_query"]
        session_id = processed_input["session_id"]

        chat_context : ChatContextManager = dependencies["chat_context"]

        chat_context.add_message(session_id, role="user", content=user_query)

        conversation_history = chat_context.get_messages_for_session(session_id)
        
        rag_results = dependencies["rag_manager"].query_knowledge(query_text=user_query, n_results=3)

        rag_context = (
            "\n".join(rag_results["documents"][0])
            if rag_results and rag_results["documents"]
            else "No relevant documents found."
        )

        final_prompt = dependencies["prompt_manager"].load_template(
            self.prompt_template_name,  # type: ignore
            history=conversation_history,
            context=rag_context,
            query=user_query,
        )

        core_ai_service = dependencies["core_ai_service"]  # type: ignore
        generation_params = dependencies.get("generation_params", {})

        model_response_text: str = core_ai_service.process_prompt(  # type: ignore
            final_prompt, generation_params
        )

        chat_context.add_message(
            session_id, role="assistant", content=model_response_text
        )

        return model_response_text # type: ignore
