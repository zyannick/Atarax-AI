from pathlib import Path
from typing import Dict, Any

class PromptManager:

    def __init__(self, prompts_directory: Path):
        if not prompts_directory.is_dir():
            raise FileNotFoundError(f"The specified prompts directory does not exist: {prompts_directory}")
        self.prompts_dir = prompts_directory
        self._cache: Dict[str, str] = {}
        print(f"PromptManager initialized for directory: {self.prompts_dir}")

    def load_template(self, template_name: str, **kwargs: Any) -> str:
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
        
        
        