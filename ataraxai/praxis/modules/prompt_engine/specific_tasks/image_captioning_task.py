from typing import Any, Dict
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ataraxai.praxis.modules.prompt_engine.context_manager import TaskContext
from ataraxai.praxis.modules.prompt_engine.specific_tasks.base_task import BaseTask
from typing import Optional


class ImageCaptioningTask(BaseTask):
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf"):
        """
        Initializes the image captioning task with the specified model.

        Args:
            model_id (str, optional): The identifier of the model to use for image captioning. 
                Defaults to "llava-hf/llava-1.5-7b-hf".

        Attributes:
            id (str): Unique identifier for the image captioning task.
            description (str): Description of the task.
            required_inputs (List[str]): List of required input keys for the task.
            model_id (str): The model identifier used for loading the captioning model.
            device (str): The device to run the model on ("cuda" if available, otherwise "cpu").
            model (Optional[LlavaForConditionalGeneration]): The loaded captioning model instance.
            processor (Optional[AutoProcessor]): The processor for preparing inputs for the model.
        """
        self.id = "image_captioning"
        self.description = "Generates a descriptive caption for a given image."
        self.required_inputs = ["image_path"]

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Optional[LlavaForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None

        super().__init__()

    def _load_resources(self) -> None:
        """
        Loads the image captioning model and its processor into memory.

        This method initializes the model specified by `self.model_id` onto the device specified by `self.device`,
        using half-precision (float16) and optimized memory usage settings. It also loads the corresponding processor
        for preprocessing inputs.

        Side Effects:
            - Sets `self.model` to an instance of LlavaForConditionalGeneration loaded with the specified parameters.
            - Sets `self.processor` to an instance of AutoProcessor for the same model.

        Raises:
            Any exceptions raised by `LlavaForConditionalGeneration.from_pretrained` or `AutoProcessor.from_pretrained`
            if loading fails.
        """
        print(
            f"ImageCaptioningTask: Loading model '{self.model_id}' to device '{self.device}'..."
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(  # type: ignore
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)  # type: ignore

    def execute(
        self,
        processed_input: Dict[str, Any],
        context: TaskContext,
        dependencies: Dict[str, Any],
    ) -> str:
        """
        Executes the image captioning task by generating a brief description of the main subjects in the provided image.

        Args:
            processed_input (Dict[str, Any]): A dictionary containing the input data. Must include the key 'image_path' with the path to the image file.
            context (TaskContext): The context object for the current task execution.
            dependencies (Dict[str, Any]): A dictionary of dependencies required for task execution.

        Returns:
            str: A one-sentence description of the main subjects in the image, or an error message if resources are unavailable.

        Raises:
            ValueError: If 'image_path' is not present in the input dictionary.
            IOError: If the image file cannot be opened or read.
        """
        image_path = processed_input.get("image_path")
        if not image_path:
            raise ValueError("Input dictionary must contain 'image_path'.")

        try:
            raw_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to open or read the image file at {image_path}: {e}")

        prompt = "USER: <image>\nWhat are the main subjects in this image? Provide a brief, one-sentence description.\nASSISTANT:"

        if self.processor and self.model:
            inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.device, torch.float16)  # type: ignore
            generate_ids = self.model.generate(**inputs, max_new_tokens=75)  # type: ignore
            result: str = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]  # type: ignore

            assistant_response = str(result.split("ASSISTANT:")[-1].strip())  # type: ignore

            return assistant_response
        else:
            return "Image processing resources are not available."
