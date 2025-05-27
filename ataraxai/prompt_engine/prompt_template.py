


class PromptTemplate:


    def __init__(self, template: str):
        self.template = template

    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs) if kwargs else self.template
    

    def __str__(self) -> str:
        return self.template
    
    def __repr__(self) -> str:
        return f"PromptTemplate({self.template})"