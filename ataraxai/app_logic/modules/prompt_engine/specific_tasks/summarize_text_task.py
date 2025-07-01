from ataraxai.prompt_engine.base_task import BaseTask

class SummarizeTextTask(BaseTask):
    
    def __init__(self):
        self.id = "summarize_text"
        self.description = "Summarizes a given text."
        self.required_inputs = ["text"]
        self.prompt_template_name = "summarize_text"
        super().__init__()

