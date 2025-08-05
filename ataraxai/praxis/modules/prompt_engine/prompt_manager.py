from pathlib import Path
from typing import Dict, Any, List
import logging
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.configs.config_schemas.rag_config_schema import RAGConfig
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager


class PromptManager:

    def __init__(
        self,
        prompts_directory: Path,
        logger: logging.Logger = AtaraxAILogger("PromptManager").get_logger(),
    ):
        if not prompts_directory.is_dir():
            raise FileNotFoundError(
                f"The specified prompts directory does not exist: {prompts_directory}"
            )
        self.logger = logger
        self.prompts_dir = prompts_directory
        self._cache: Dict[str, str] = {}
        self.logger.info(f"PromptManager initialized for directory: {self.prompts_dir}")

    def build_prompt_within_limit(
        self,
        history: List[Dict[str, Any]],
        rag_context: str,
        user_query: str,
        prompt_template: str,
        context_limit: int,
        core_ai_service_manager: CoreAIServiceManager,
        rag_config: RAGConfig,
    ) -> str:
        try:
            query_tokens = core_ai_service_manager.tokenize(user_query)
            template_shell = (
                prompt_template.replace("{history}", "")
                .replace("{context}", "")
                .replace("{query}", "")
            )
            template_tokens = core_ai_service_manager.tokenize(template_shell)

            generation_params = (
                core_ai_service_manager.config_manager.llama_config_manager.get_generation_params()
            )
            prompt_budget = context_limit - generation_params.n_predict
            total_content_budget = prompt_budget - (
                len(query_tokens) + len(template_tokens)
            )

            if total_content_budget <= 0:
                self.logger.warning(
                    "No budget available for history and context after accounting for query and template"
                )
                return prompt_template.format(history="", context="", query=user_query)

            rag_budget = int(total_content_budget * rag_config.context_allocation_ratio)
            history_budget = total_content_budget - rag_budget

            truncated_rag_context = self._truncate_text_to_budget(
                rag_context, rag_budget, core_ai_service_manager
            )

            final_history_str = self._build_history_within_budget(
                history, history_budget, core_ai_service_manager
            )

            if not final_history_str and not truncated_rag_context:
                self.logger.warning(
                    "Unable to fit any history or context within token budget"
                )
                truncated_rag_context = "No relevant documents found."

            return prompt_template.format(
                history=final_history_str,
                context=truncated_rag_context,
                query=user_query,
            )

        except Exception as e:
            self.logger.error(f"Error building prompt within limit: {e}")
            return prompt_template.format(history="", context="", query=user_query)

    def _truncate_text_to_budget(
        self, text: str, budget: int, core_ai_service_manager: CoreAIServiceManager
    ) -> str:
        if not text or budget <= 0:
            return ""

        tokens = core_ai_service_manager.tokenize(text)
        if len(tokens) <= budget:
            return text

        truncated_tokens = tokens[:budget]
        return core_ai_service_manager.decode(truncated_tokens)

    def _build_history_within_budget(
        self,
        history: List[Dict[str, Any]],
        budget: int,
        core_ai_service_manager: CoreAIServiceManager,
    ) -> str:
        if budget <= 0:
            return ""

        final_history_str = ""
        current_tokens = 0

        for message in reversed(history):
            try:
                if (
                    not isinstance(message, dict)
                    or "role" not in message
                    or "content" not in message
                ):
                    self.logger.warning(f"Skipping invalid message format: {message}")
                    continue

                message_str = f"{message['role']}: {message['content']}\n"
                message_tokens = core_ai_service_manager.tokenize(message_str)

                if current_tokens + len(message_tokens) <= budget:
                    final_history_str = message_str + final_history_str
                    current_tokens += len(message_tokens)
                else:
                    remaining_budget = budget - current_tokens
                    if remaining_budget > 50:
                        role_prefix = f"{message['role']}: "
                        prefix_tokens = len(
                            core_ai_service_manager.tokenize(role_prefix)
                        )
                        content_budget = remaining_budget - prefix_tokens - 1

                        if content_budget > 0:
                            truncated_content = self._truncate_text_to_budget(
                                message["content"],
                                content_budget,
                                core_ai_service_manager,
                            )
                            if truncated_content:
                                truncated_message = (
                                    f"{role_prefix}{truncated_content}...\n"
                                )
                                final_history_str = (
                                    truncated_message + final_history_str
                                )
                    break

            except Exception as e:
                self.logger.warning(f"Error processing message {message}: {e}")
                continue

        return final_history_str

    def load_template(self, template_name: str, **kwargs: Any) -> str:
        if template_name not in self._cache:
            prompt_path = self.prompts_dir / f"{template_name}.txt"
            if not prompt_path.exists():
                raise FileNotFoundError(
                    f"Prompt template '{template_name}' not found at {prompt_path}"
                )

            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    self._cache[template_name] = f.read()
                self.logger.debug(f"Loaded template '{template_name}' into cache")
            except Exception as e:
                raise IOError(f"Error reading template file {prompt_path}: {e}")

        template = self._cache[template_name]

        try:
            return template.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'\"")
            self.logger.warning(
                f"Missing placeholder '{missing_key}' for template '{template_name}'. Returning unformatted template."
            )
            return template
        except Exception as e:
            self.logger.error(f"Error formatting template '{template_name}': {e}")
            return template

    def clear_cache(self) -> None:
        self._cache.clear()
        self.logger.info("Template cache cleared")

    def get_cached_templates(self) -> List[str]:
        return list(self._cache.keys())

    def template_exists(self, template_name: str) -> bool:
        prompt_path = self.prompts_dir / f"{template_name}.txt"
        return prompt_path.exists()

    def list_available_templates(self) -> List[str]:
        try:
            return [f.stem for f in self.prompts_dir.glob("*.txt")]
        except Exception as e:
            self.logger.error(f"Error listing templates: {e}")
            return []
