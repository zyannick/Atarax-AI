from pathlib import Path
from typing import Dict, Any

class PromptManager:

    def __init__(self, prompts_directory: Path):
        """
        Initializes the PromptManager with the specified prompts directory.

        Args:
            prompts_directory (Path): The path to the directory containing prompt files.

        Raises:
            FileNotFoundError: If the specified prompts directory does not exist.

        Attributes:
            prompts_dir (Path): The directory containing prompt files.
            _cache (Dict[str, str]): Internal cache for loaded prompts.

        Prints:
            Confirmation message indicating the initialized prompts directory.
        """
        if not prompts_directory.is_dir():
            raise FileNotFoundError(f"The specified prompts directory does not exist: {prompts_directory}")
        self.prompts_dir = prompts_directory
        self._cache: Dict[str, str] = {}
        print(f"PromptManager initialized for directory: {self.prompts_dir}")

    def load_template(self, template_name: str, **kwargs: Any) -> str:
        """
        Loads a prompt template by name, formats it with provided keyword arguments, and caches it for future use.
        Args:
            template_name (str): The name of the template file (without extension) to load from the prompts directory.
            **kwargs (Any): Keyword arguments to format the template with.
        Returns:
            str: The formatted template string if all placeholders are provided, otherwise the raw template string.
        Raises:
            FileNotFoundError: If the template file does not exist in the prompts directory.
        Notes:
            - If a required placeholder is missing in kwargs, a warning is printed and the unformatted template is returned.
            - Loaded templates are cached to avoid repeated file reads.
        """
        if template_name not in self._cache:
            prompt_path = self.prompts_dir / f"{template_name}.txt"
            
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt template '{template_name}' not found at {prompt_path}")
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self._cache[template_name] = f.read()
        
        template = self._cache[template_name]
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"Warning: Placeholder {e} not provided for template '{template_name}'.")
            return template
        
        
        