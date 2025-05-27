from ataraxai.prompt_engine.base_task import BaseTask


class OCRandSummarizeTask(BaseTask):

    def __init__(self):
        self.id = "ocr_and_summarize"
        self.description = "Extracts text from images and summarizes the content."
        self.required_inputs = ["image"]
        self.prompt_template_name = "ocr_and_summarize"
        super().__init__()
    
    def execute(self, image_data: dict):

        pass