from typing import Any, Dict
from PIL import Image
import pytesseract

from ataraxai.app_logic.modules.prompt_engine.context_manager import TaskContext
from ataraxai.app_logic.modules.prompt_engine.prompt_manager import PromptManager
from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask
from typing import List

from ataraxai import core_ai_py  # type: ignore


class OCRandSummarizeTask(BaseTask):
    def __init__(self):
        self.id: str = "ocr_and_summarize"
        self.description: str = (
            "Extracts text from an image using OCR and then summarizes it."
        )
        self.required_inputs: List[str] = ["image_path"]
        self.prompt_template_name: str = "ocr_summarize"
        super().__init__()

    def _load_resources(self) -> None:
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            print(f"Tesseract OCR engine found. Version: {tesseract_version}")
        except pytesseract.TesseractNotFoundError:
            print("ERROR: Tesseract OCR engine not found.")
            print("Please install it on your system and ensure it's in your PATH.")
            raise

    def execute(
        self,
        processed_input: Dict[str, Any],
        context: TaskContext,
        dependencies: Dict[str, Any],
    ) -> str:
        image_path = processed_input.get("image_path")
        if not image_path:
            raise ValueError("Input dictionary must contain 'image_path'.")

        print(f"Running OCR on image: {image_path}")
        try:
            image = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(image)  # type: ignore
        except Exception as e:
            raise IOError(f"Failed to process image with OCR: {e}")

        if not extracted_text or not extracted_text.strip():  # type: ignore
            return "No text could be extracted from the image."

        print(f"Extracted {len(extracted_text)} characters from image.")

        print("Summarizing extracted text...")
        try:
            summarization_prompt_template = dependencies[
                "prompt_manager"
            ].load_template(self.prompt_template_name)
            final_prompt = summarization_prompt_template.format(ocr_text=extracted_text)
            summary: str = dependencies["core_ai_service"].process_prompt(  # type: ignore
                final_prompt, dependencies["generation_params"]
            )

            return summary  # type: ignore

        except Exception as e:
            raise RuntimeError(f"Failed to generate summary with LLM: {e}")
