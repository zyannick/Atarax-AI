from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask

from typing import Any, Dict

from ataraxai.app_logic.modules.prompt_engine.context_manager import TaskContext
from ataraxai.app_logic.modules.prompt_engine.prompt_manager import PromptManager
from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask
from typing import List
from ataraxai import core_ai_py  # type: ignore

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
        prompt_manager: PromptManager,
        core_ai_service: core_ai_py.CoreAiService,  # type: ignore
        generation_params: Dict[str, Any]
    ) -> str:
        text_to_summarize = processed_input.get("text")
        if not text_to_summarize:
            raise ValueError("Input dictionary must contain 'text'.")

        prompt_template = prompt_manager.load_template(
            self.prompt_template_name,
            text_to_summarize=text_to_summarize
        )

        print(f"Generating summary for text of length {len(text_to_summarize)}...")
        summary = core_ai_service.process_prompt(prompt_template, generation_params) # type: ignore

        
        return summary.strip() # type: ignore