from ataraxai.praxis.modules.prompt_engine.specific_tasks.base_task import BaseTask

from typing import Any, Dict, Optional

from ataraxai.praxis.modules.prompt_engine.context_manager import TaskContext
from typing import List
from ataraxai.praxis.modules.prompt_engine.specific_tasks.task_dependencies import TaskDependencies


class SummarizeTextTask(BaseTask):

    def __init__(self):
        """
        Initializes the SummarizeTextTask instance with default values.

        Attributes:
            id (str): Unique identifier for the task.
            description (str): Brief description of the task.
            required_inputs (List[str]): List of required input keys for the task.
            prompt_template_name (str): Name of the prompt template to use.
        """
        self.id: str = "summarize_text"
        self.description: str = "Summarizes a given block of text."
        self.required_inputs: List[str] = ["text"]
        self.prompt_template_name: Optional[str] = "summarize_text"
        super().__init__()

    def _load_resources(self) -> None:
        """
        Loads any special resources required for the task.

        This implementation does not load any resources, as none are needed for this task.
        Prints a message indicating that no special resources are required.
        """
        print(f"Task '{self.id}' requires no special resources to load.")
        pass

    def execute(
        self,
        processed_input: Dict[str, Any],
        context: TaskContext,
        dependencies: TaskDependencies,
    ) -> str:
        """
        Executes the text summarization task using the provided input, context, and dependencies.
        Args:
            processed_input (Dict[str, Any]): Dictionary containing the input data. Must include the key 'text' with the text to summarize.
            context (TaskContext): The context object for the current task execution.
            dependencies (Dict[str, Any]): Dictionary of dependencies required for execution, including:
                - 'prompt_manager': An object responsible for loading prompt templates.
                - 'core_ai_service': The AI service used to generate the summary.
                - 'generation_params' (optional): Parameters for the AI generation process.
        Returns:
            str: The summarized text.
        Raises:
            ValueError: If 'text' is not present in the processed_input dictionary.
        """
        text_to_summarize = processed_input.get("text")
        if not text_to_summarize:
            raise ValueError("Input dictionary must contain 'text'.")

        prompt_template = dependencies["prompt_manager"].load_template(
            self.prompt_template_name, text_to_summarize=text_to_summarize
        )

        core_ai_service = dependencies["core_ai_service"]  # type: ignore
        generation_params = dependencies.get("generation_params", {})

        print(f"Generating summary for text of length {len(text_to_summarize)}...")
        summary = core_ai_service.process_prompt(prompt_template, generation_params)  # type: ignore

        return summary.strip()
