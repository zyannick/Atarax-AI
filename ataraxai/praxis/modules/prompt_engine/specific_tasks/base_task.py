from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from ataraxai.praxis.modules.prompt_engine.context_manager import TaskContext


class BaseTask(ABC):
    id: str
    description: str
    required_inputs: List[str] = []
    prompt_template_name: Optional[str] = None

    def __init__(self):
        """
        Initializes the base task instance.

        Ensures that concrete subclasses define both an 'id' and a 'description' attribute.
        Raises:
            NotImplementedError: If 'id' or 'description' is not defined in the subclass.
        """
        self._initialized: bool = False
        if not hasattr(self, "id") or not self.id:
            raise NotImplementedError("Concrete tasks must define an 'id'")
        if not hasattr(self, "description") or not self.description:
            raise NotImplementedError("Concrete tasks must define a 'description'")

    def load_if_needed(self):
        """
        Lazily loads resources required for the task if they have not been initialized yet.

        This method checks whether the task's resources have already been loaded. If not,
        it performs the loading process by calling the internal `_load_resources` method,
        marks the task as initialized, and prints status messages before and after loading.

        Returns:
            None
        """
        if not self._initialized:
            print(f"Lazy loading resources for task: '{self.id}'...")
            self._load_resources()
            self._initialized = True
            print(f"Resources for '{self.id}' loaded successfully.")

    def run(
        self,
        input_data: Dict[str, Any],
        context: TaskContext,
        dependencies: Dict[str, Any],
    ) -> Any:
        """
        Executes the main logic of the task, handling input validation, preprocessing, execution, and postprocessing steps.

        Args:
            input_data (Dict[str, Any]): The input data required for the task.
            context (TaskContext): The context object containing task-specific information.
            dependencies (Dict[str, Any]): A dictionary of dependencies required for task execution.

        Returns:
            Any: The final processed output of the task.

        Raises:
            Exception: Any exception raised during the execution is handled by the handle_error method.
        """
        try:
            self.load_if_needed()
            self.validate_inputs(input_data)
            processed_input = self.preprocess(input_data, context)
            raw_output = self.execute(processed_input, context, dependencies)
            return self.postprocess(raw_output, context)
        except Exception as e:
            return self.handle_error(e, context)

    @abstractmethod
    def _load_resources(self) -> None:
        pass

    def validate_inputs(self, input_data: Dict[str, Any]) -> None:
        missing = [inp for inp in self.required_inputs if inp not in input_data]
        if missing:
            raise ValueError(f"Task '{self.id}' missing required inputs: {missing}")

    def handle_error(self, error: Exception, context: TaskContext) -> Any:
        print(f"ERROR during execution of task '{self.id}': {error}")
        raise error

    def preprocess(
        self, input_data: Dict[str, Any], context: TaskContext
    ) -> Dict[str, Any]:
        return input_data

    @abstractmethod
    def execute(
        self,
        processed_input: Dict[str, Any],
        context: TaskContext,
        dependencies: Dict[str, Any],
    ) -> Any:
        pass

    def postprocess(self, raw_output: Any, context: TaskContext) -> Any:
        return raw_output

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "required_inputs": self.required_inputs,
            "prompt_template": self.prompt_template_name,
        }
