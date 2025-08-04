from typing import Any, Dict
from PIL import Image
import pytesseract

from ataraxai.praxis.modules.prompt_engine.specific_tasks.base_task import BaseTask
from typing import List

from ataraxai.praxis.modules.prompt_engine.specific_tasks.task_dependencies import TaskDependencies


class OCRandSummarizeTask(BaseTask):
    def __init__(self):
        """
        Initializes the OCR and Summarize task with predefined attributes.

        Attributes:
            id (str): Unique identifier for the task.
            description (str): Brief description of the task's functionality.
            required_inputs (List[str]): List of required input parameter names.
            prompt_template_name (str): Name of the prompt template to use.
        """
        self.id: str = "ocr_and_summarize"
        self.description: str = (
            "Extracts text from an image using OCR and then summarizes it."
        )
        self.required_inputs: List[str] = ["image_path"]
        self.prompt_template_name: str = "ocr_summarize"
        super().__init__()

    def _load_resources(self) -> None:
        """
        Checks for the presence of the Tesseract OCR engine and prints its version.

        Attempts to retrieve and display the installed Tesseract OCR engine version.
        If Tesseract is not found, prints an error message and raises an exception.

        Raises:
            pytesseract.TesseractNotFoundError: If the Tesseract OCR engine is not installed or not found in the system PATH.
        """
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
        dependencies: TaskDependencies,
    ) -> str:
        """
        Executes the OCR and summarization task on a given image.

        Args:
            processed_input (Dict[str, Any]): Dictionary containing the input data. Must include the key 'image_path' with the path to the image file.
            context (TaskContext): The context object for the current task execution.
            dependencies (Dict[str, Any]): Dictionary of dependencies required for execution, including 'prompt_manager' and 'core_ai_service'.

        Returns:
            str: The summary generated from the extracted text in the image, or a message indicating no text was extracted.

        Raises:
            ValueError: If 'image_path' is not present in processed_input.
            IOError: If OCR processing fails on the image.
            RuntimeError: If summarization with the language model fails.
        """
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

        prompt_manager = dependencies["prompt_manager"]

        print("Summarizing extracted text...")
        try:
            final_prompt = prompt_manager.load_template(
                self.prompt_template_name, ocr_text=extracted_text
            )

            summary: str = dependencies["core_ai_service"].process_prompt(  # type: ignore
                final_prompt, dependencies["generation_params"]
            )

            return summary  # type: ignore

        except Exception as e:
            raise RuntimeError(f"Failed to generate summary with LLM: {e}")
