from ataraxai.prompt_engine.base_task import BaseTask

class AnswerFromLocalDocsTask(BaseTask):

    def __init__(self):
        self.id = "answer_from_local_docs"
        self.description = "Answers a question using information from local documents."
        self.required_inputs = ["question", "local_documents"]
        self.prompt_template_name = "answer_from_local_docs"
        super().__init__()