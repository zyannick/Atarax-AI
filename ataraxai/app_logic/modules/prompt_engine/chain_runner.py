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
        """
        Initializes the ChainRunner with required managers and services.

        Args:
            task_manager (TaskManager): Manages tasks within the chain.
            context_manager (ContextManager): Handles context-related operations.
            prompt_manager (PromptManager): Manages prompt templates and logic.
            core_ai_service (Any): Core AI service used for processing (type ignored for flexibility).
            chat_context (ChatContextManager): Manages chat-specific context and state.
            rag_manager (AtaraxAIRAGManager): Handles retrieval-augmented generation (RAG) operations.

        Attributes:
            task_manager (TaskManager): Reference to the task manager.
            context_manager (ContextManager): Reference to the context manager.
            prompt_manager (PromptManager): Reference to the prompt manager.
            core_ai_service (Any): Reference to the core AI service.
            chat_context (ChatContextManager): Reference to the chat context manager.
            rag_manager (AtaraxAIRAGManager): Reference to the RAG manager.
            dependencies (Dict[str, Any]): Dictionary of dependencies for internal use.
        """
        self.task_manager = task_manager
        self.context_manager = context_manager
        self.prompt_manager = prompt_manager
        self.core_ai_service = core_ai_service
        self.chat_context = chat_context
        self.rag_manager = rag_manager

        self.dependencies: Dict[str, Any] = {
            "context_manager": self.context_manager,
            "prompt_manager": self.prompt_manager,
            "core_ai_service": self.core_ai_service,
            "chat_context": self.chat_context,
            "rag_manager": self.rag_manager
        }

    def run_chain(
        self, chain_definition: List[Dict[str, Any]], initial_user_query: str
    ) -> Any:
        """
        Executes a sequence of tasks defined in a chain, passing outputs from previous steps as inputs to subsequent steps.

        Args:
            chain_definition (List[Dict[str, Any]]): 
                A list of dictionaries, each representing a step in the chain. Each step must specify a 'task_id' and may specify 'inputs', 
                where input values can reference outputs from previous steps using the format '{{step_n.key}}'.
            initial_user_query (str): 
                The initial user query to be used as context for the chain execution.

        Returns:
            Any: 
                The output of the final executed task in the chain, or the result of error handling if an exception occurs.

        Raises:
            Exception: 
                If a task raises an exception and does not handle it internally, the exception is caught, error handling is invoked, and execution stops.
        """
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
