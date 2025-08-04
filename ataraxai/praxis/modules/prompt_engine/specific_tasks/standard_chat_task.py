from typing import Any, Dict

from ataraxai.praxis.modules.prompt_engine.specific_tasks.base_task import BaseTask
from ataraxai.praxis.modules.prompt_engine.specific_tasks.task_dependencies import (
    TaskDependencies,
)


class StandardChatTask(BaseTask):
    def __init__(self):
        self.id = "standard_chat"
        self.description = "Performs a standard chat interaction with RAG context."
        self.required_inputs = ["user_query", "session_id"]
        self.prompt_template_name = "standard_chat"
        super().__init__()

    def _load_resources(self) -> None:
        pass

    def execute(
        self,
        processed_input: Dict[str, Any],
        dependencies: TaskDependencies,
    ) -> str:

        user_query = processed_input["user_query"]
        session_id = processed_input["session_id"]

        chat_context = dependencies["chat_context"]
        rag_manager = dependencies["rag_manager"]
        prompt_manager = dependencies["prompt_manager"]
        core_ai_service_manager = dependencies["core_ai_service_manager"]
        context_manager = dependencies["context_manager"]

        model_context_limit = core_ai_service_manager.get_llama_cpp_model_context_size()

        chat_context.add_message(session_id, role="user", content=user_query)
        session_history = chat_context.get_messages_for_session(session_id)

        rag_results = context_manager.get_context(context_key="relevant_document_chunks", user_inputs=user_query)
        if not rag_results:
            rag_context_str = ""
        else:
            rag_context_str = "\n".join(rag_results)

        prompt_template_str = prompt_manager.load_template(self.prompt_template_name)

        final_prompt = prompt_manager.build_prompt_within_limit(
            history=session_history,
            rag_context=rag_context_str,
            user_query=user_query,
            prompt_template=prompt_template_str,
            context_limit=model_context_limit,
            core_ai_service_manager=core_ai_service_manager,
            rag_config=rag_manager.rag_config_manager.config,
        )

        model_response_text = core_ai_service_manager.process_prompt(final_prompt)

        assistant_response = model_response_text.strip()
        if not assistant_response:
            assistant_response = "I'm sorry, I couldn't generate a response."
        else:
            assistant_response = assistant_response.split("assistant:")[-1].strip()

        chat_context.add_message(
            session_id, role="assistant", content=assistant_response
        )

        return assistant_response
