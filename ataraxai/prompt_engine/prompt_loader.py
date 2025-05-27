


class PromptLoader:
    def __init__(self, prompt_path: str):
        self.prompt_path = prompt_path

    def load_template(self) -> str:
        """
        Load the prompt template from the specified file path.
        """
        return self.load_prompt()

    def load_prompt(self) -> str:
        with open(self.prompt_path, 'r') as file:
            prompt = file.read()
        return prompt