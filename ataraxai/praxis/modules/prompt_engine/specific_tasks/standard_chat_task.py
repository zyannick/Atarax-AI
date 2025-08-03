from typing import Any, Dict, List

from ataraxai.praxis.modules.prompt_engine.specific_tasks.base_task import BaseTask
from ataraxai.praxis.modules.prompt_engine.context_manager import TaskContext
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.praxis.modules.prompt_engine.specific_tasks.task_dependencies import TaskDependencies
from ataraxai.praxis.utils.configs.config_schemas.rag_config_schema import RAGConfig
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager


class StandardChatTask(BaseTask):
    def __init__(self):
        self.id = "standard_chat"
        self.description = "Performs a standard chat interaction with RAG context."
        self.required_inputs = ["user_query", "session_id"]
        self.prompt_template_name = "standard_chat"
        super().__init__()

    def _load_resources(self) -> None:
        pass  

    def _build_prompt_within_limit(
        self,
        history: List[Dict[str, Any]],
        rag_context: str,
        user_query: str,
        prompt_template: str,
        context_limit: int,
        core_ai_service_manager: CoreAIServiceManager,
        rag_config: RAGConfig,
    ) -> str:
        # query_tokens = core_ai_service_manager.tokenize(user_query)
        # template_shell = prompt_template.replace("{history}", "").replace("{context}", "").replace("{query}", "")
        # template_tokens = core_ai_service_manager.tokenize(template_shell)

        # prompt_budget = context_limit - core_ai_service_manager.config_manager.llama_config_manager.get_generation_params().n_predict
        # total_content_budget = prompt_budget - (len(query_tokens) + len(template_tokens))
        # rag_budget = int(total_content_budget * rag_config.context_allocation_ratio)
        # history_budget = total_content_budget - rag_budget

        # rag_tokens = core_ai_service_manager.tokenize(rag_context)
        # if len(rag_tokens) > rag_budget:
        #     truncated_rag_tokens = rag_tokens[:rag_budget]
        #     rag_context = core_ai_service_manager.decode(truncated_rag_tokens)

        # final_history_str = ""
        # current_history_tokens = 0
        # for message in reversed(history):
        #     message_str = f"{message['role']}: {message['content']}\n"
        #     message_tokens = core_ai_service_manager.tokenize(message_str)
            
        #     if current_history_tokens + len(message_tokens) <= history_budget:
        #         final_history_str = message_str + final_history_str
        #         current_history_tokens += len(message_tokens)
        #     else:
        #         break

        final_history_str = ""
        rag_context = "No relevant documents found."


        return prompt_template.format(
            history=final_history_str,
            context=rag_context,
            query=user_query,
        )

    def execute(
        self,
        processed_input: Dict[str, Any],
        context: TaskContext,
        dependencies: TaskDependencies,
    ) -> str:

        user_query = processed_input["user_query"]
        session_id = processed_input["session_id"]

        chat_context = dependencies["chat_context"]
        rag_manager = dependencies["rag_manager"]
        prompt_manager = dependencies["prompt_manager"]
        core_ai_service_manager = dependencies["core_ai_service_manager"]
        
        model_context_limit = core_ai_service_manager.get_llama_cpp_model_context_size()

        chat_context.add_message(session_id, role="user", content=user_query)
        session_history = chat_context.get_messages_for_session(session_id)

        rag_results = rag_manager.query_knowledge(query_text=user_query)
        rag_context_str = "\n".join(rag_results["documents"][0]) if rag_results and rag_results["documents"] else "No relevant documents found."

        prompt_template_str = prompt_manager.load_template(self.prompt_template_name)

        final_prompt = self._build_prompt_within_limit(
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

        chat_context.add_message(session_id, role="assistant", content=assistant_response)

        return assistant_response