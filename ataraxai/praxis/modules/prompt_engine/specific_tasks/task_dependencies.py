from typing import TypedDict, Any
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager
from ataraxai.praxis.modules.prompt_engine.context_manager import ContextManager
from ataraxai.praxis.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager
from ataraxai.praxis.modules.prompt_engine.prompt_manager import PromptManager
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager

class TaskDependencies(TypedDict):
    chat_context: ChatContextManager
    rag_manager: AtaraxAIRAGManager
    prompt_manager: PromptManager
    context_manager: ContextManager
    chat_context: ChatContextManager
    core_ai_service_manager: CoreAIServiceManager 