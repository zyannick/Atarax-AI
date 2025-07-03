from typing import Any, Dict, List
from .context_manager import ContextManager, TaskContext
from ataraxai.app_logic.modules.prompt_engine.prompt_manager import PromptManager
from ataraxai.app_logic.modules.prompt_engine.task_manager import TaskManager
from ataraxai.app_logic.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager
from ataraxai.app_logic.modules.chat.chat_context_manager import ChatContextManager


class ChainRunner:
    def __init__(
        self,
        task_manager: TaskManager,
        context_manager: ContextManager,
        prompt_manager: PromptManager,
        core_ai_service: Any,  # type: ignore
        chat_context: ChatContextManager,
        rag_manager: AtaraxAIRAGManager
    ):
        self.task_manager = task_manager
        self.context_manager = context_manager
        self.prompt_manager = prompt_manager
        self.core_ai_service = core_ai_service
        self.chat_context = chat_context
        self.rag_manager = rag_manager

        self.dependencies: Dict[str, Any] = {
            "context_manager": context_manager,
            "prompt_manager": prompt_manager,
            "core_ai_service": core_ai_service,
            "chat_context": chat_context,
            "rag_manager": rag_manager
        }

    def run_chain(
        self, chain_definition: List[Dict[str, Any]], initial_user_query: str
    ) -> Any:
        context = TaskContext(user_query=initial_user_query)
        step_outputs: Dict[str, Any] = {}
        final_result = None

        for i, step in enumerate(chain_definition):
            task_id = step["task_id"]
            task = self.task_manager.get_task(task_id)

            input_data = {}
            for key, value in step.get("inputs", {}).items():
                if (
                    isinstance(value, str)
                    and value.startswith("{{")
                    and value.endswith("}}")
                ):
                    ref_step, ref_key = value.strip(" {}").split(".")
                    input_data[key] = step_outputs[ref_step][ref_key]
                else:
                    input_data[key] = value

            print(f"\n--- Running Step {i}: Task '{task_id}' ---")

            try:
                final_result = task.run(input_data, context, self.dependencies)

                step_outputs[f"step_{i}"] = {"output": final_result}
                print(
                    f"--- Step {i} Successful. Result: {str(final_result)[:100]}... ---"
                )

            except Exception as e:
                print(f"--- Error in Step {i}, Task '{task_id}' ---")
                final_result = task.handle_error(e, context)
                break

        return final_result
