from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict


from ..prompt_loader import PromptLoader
from ..context_manager import TaskContext 



class BaseTask(ABC):
    id: str 
    description: str 
    required_inputs: List[str] = []
    prompt_template_name: Optional[str] = None

    def __init__(self):
        if not hasattr(self, 'id') or not self.id:
            raise NotImplementedError("Concrete tasks must define an 'id'")
        if not hasattr(self, 'description') or not self.description:
            raise NotImplementedError("Concrete tasks must define a 'description'")

    def preprocess(self, input_data: Dict[str, Any], context: TaskContext) -> Dict[str, Any]:
        return input_data

    @abstractmethod
    def execute(
        self,
        processed_input: Dict[str, Any],
        context: TaskContext,
        prompt_loader: PromptLoader,
    ) -> Any:
        pass

    def postprocess(self, raw_output: Any, context: TaskContext) -> Any:
        return raw_output
