from ataraxai import core_ai_py
from ataraxai.core_ai_py import CoreAIService
from ataraxai.app_logic.preferences_manager import PreferencesManager
from ataraxai.app_logic.modules.rag.resilient_indexer import start_rag_file_monitoring
import os


def starter():
    """
    Initialize the AtaraxAI application.
    """
    # Load preferences
    user_home = os.getenv("HOME") or os.getenv("USERPROFILE")
    if not user_home:
        raise EnvironmentError(
            "Could not determine user home directory. Please set HOME or USERPROFILE environment variable."
        )
    else:
        print(f"User home directory: {user_home}")

    prefs_manager = PreferencesManager()

    # Get watched directories from preferences
    watched_directories = prefs_manager.get("watched_directories", [])
    if os.path.join(user_home, "Documents") not in watched_directories:
        watched_directories.append(os.path.join(user_home, "Documents"))
        prefs_manager.set("watched_directories", watched_directories)
        print(
            f"Added default Documents directory to watched directories: {os.path.join(user_home, 'Documents')}"
        )
    else:
        print(
            f"Watched directories already include Documents: {os.path.join(user_home, 'Documents')}"
        )

    core_ai_service = core_ai_py.CoreAIService()
    print(core_ai_service)
    print("AtaraxAI initialized successfully.")

    llama_params = core_ai_py.LlamaModelParams(
        model_path="/home/yzoetgna/projects/Atarax-AI/data/last_models/models/llama/Qwen3-30B-A3B-UD-IQ1_S.gguf",
        n_ctx=8192,
        n_gpu_layers=32,
        main_gpu=0,
        tensor_split=False,
        vocab_only=False,
        use_map=False,
        use_mlock=False,
    )
    
    print("Llama model parameters set:", llama_params)

    # # Initialize core AI components
    # core_ai_py.initialize()

    # Start file monitoring for RAG updates
    # if watched_directories:
    #     observer = start_rag_file_monitoring(watched_directories, core_ai_py.manifest, core_ai_py.chroma_collection)
    #     print("File monitoring started.")
    # else:
    #     print("No directories to watch. Please set your preferences.")


if __name__ == "__main__":
    try:
        starter()
    except Exception as e:
        print(f"Error starting AtaraxAI: {e}")
        raise
