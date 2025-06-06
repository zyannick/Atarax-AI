from ataraxai import core_ai_py
from ataraxai.core_ai_py import CoreAIService
from pathlib import Path
from ataraxai.app_logic.prompt_utils import create_prompt

# from ataraxai.app_logic.preferences_manager import PreferencesManager
# from ataraxai.app_logic.modules.rag.resilient_indexer import start_rag_file_monitoring
import os


def starter():
    """
    Initialize the AtaraxAI application.
    """
    # Load preferences
    # user_home = os.getenv("HOME") or os.getenv("USERPROFILE")
    # if not user_home:
    #     raise EnvironmentError(
    #         "Could not determine user home directory. Please set HOME or USERPROFILE environment variable."
    #     )
    # else:
    #     print(f"User home directory: {user_home}")

    # prefs_manager = PreferencesManager()

    # # Get watched directories from preferences
    # watched_directories = prefs_manager.get("watched_directories", [])
    # if os.path.join(user_home, "Documents") not in watched_directories:
    #     watched_directories.append(os.path.join(user_home, "Documents"))
    #     prefs_manager.set("watched_directories", watched_directories)
    #     print(
    #         f"Added default Documents directory to watched directories: {os.path.join(user_home, 'Documents')}"
    #     )
    # else:
    #     print(
    #         f"Watched directories already include Documents: {os.path.join(user_home, 'Documents')}"
    #     )

    project_dir = Path(__file__).resolve().parent.parent.parent
    print(project_dir)

    core_ai_service = core_ai_py.CoreAIService()
    print(core_ai_service)
    print("AtaraxAI initialized successfully.")

    llama_params = core_ai_py.LlamaModelParams(
        model_path=str(
            project_dir / "data/last_models/models/llama/Qwen3-30B-A3B-UD-IQ1_S.gguf"
        ),
        n_ctx=8192,
        n_gpu_layers=32,
        main_gpu=0,
        tensor_split=False,
        vocab_only=False,
        use_map=False,
        use_mlock=False,
    )

    llama_generation_params = core_ai_py.GenerationParams(
        n_predict=128,
        temp=0.1,  
        top_k=20,
        top_p=0.95,
        repeat_penalty=1.2, 
        penalty_last_n=64,
        penalty_freq=0.7,  
        penalty_present=0.0,  
        stop_sequences=["</s>", "\n\n", "User:"],  
        n_batch=1,
        n_threads=4,
    )

    core_ai_service.initialize_llama_model(llama_params)

    whisper_params = core_ai_py.WhisperModelParams(
        model_path=str(project_dir / "data/last_models/models/whisper/ggml-base.en.bin"),
        n_threads=4,
        n_gpu_layers=0,
        main_gpu=0,
        tensor_split=False,
        use_map=False,
        use_mlock=False,
    )

    while True:
        default_system_message = (
            "You are a helpful AI assistant. Provide clear and concise answers."
        )

        user_input_text = input("Enter your query (or type 'exit' to quit): ")
        if user_input_text.lower() == "exit":
            print("Exiting AtaraxAI.")
            break

        full_prompt = create_prompt(
            user_query=user_input_text, system_message=default_system_message
        )
        current_generation_params = llama_generation_params
        answer = core_ai_service.process_prompt(full_prompt, current_generation_params)
        print(f"AtaraxAI assistant: {answer}")


if __name__ == "__main__":
    try:
        starter()
    except KeyboardInterrupt:
        print("AtaraxAI startup interrupted by user.")
    except Exception as e:
        print(f"Error starting AtaraxAI: {e}")
        raise
