from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask

from typing import Any, Dict

from ataraxai.app_logic.modules.prompt_engine.context_manager import TaskContext
from typing import List


class SummarizeTextTask(BaseTask):

    def __init__(self):
        self.id: str = "summarize_text"
        self.description: str = "Summarizes a given block of text."
        self.required_inputs: List[str] = ["text"]
        self.prompt_template_name: str = "summarize_text"
        super().__init__()

    def _load_resources(self) -> None:
        print(f"Task '{self.id}' requires no special resources to load.")
        pass

    def execute(
        self,
        processed_input: Dict[str, Any],
        context: TaskContext,
        dependencies: Dict[str, Any],
    ) -> str:
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
