from ataraxai.prompt_engine.base_task import BaseTask


class ImageCaptioningTask(BaseTask):

    def __init__(self):
        self.id = "image_captioning"
        self.description = "Generates a caption for a given image."
        self.required_inputs = ["image"]
        self.prompt_template_name = "image_captioning"
        super().__init__()

    def execute(self, image_data: dict):
        pass