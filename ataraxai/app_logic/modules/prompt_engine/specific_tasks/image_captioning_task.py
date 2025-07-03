from typing import Any, Dict
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ataraxai.app_logic.modules.prompt_engine.context_manager import TaskContext
from ataraxai.app_logic.modules.prompt_engine.prompt_manager import PromptManager
from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask
from typing import Optional


class ImageCaptioningTask(BaseTask):
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf"):
        self.id = "image_captioning"
        self.description = "Generates a descriptive caption for a given image."
        self.required_inputs = ["image_path"]

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Optional[LlavaForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None

        super().__init__()

    def _load_resources(self) -> None:
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
