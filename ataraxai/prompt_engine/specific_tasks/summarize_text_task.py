from ataraxai.prompt_engine.base_task import BaseTask

class SummarizeTextTask(BaseTask):
    id = "summarize_text"
    description = "Summarizes a given piece of text."
    required_inputs = ["text_to_summarize"]
    prompt_template_name = "summarize_general" # Example template name

