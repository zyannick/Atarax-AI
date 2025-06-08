from ataraxai import core_ai_py
from pathlib import Path
from ataraxai.app_logic.prompt_utils import create_prompt
from ataraxai.app_logic.utils.config_schemas.llama_config_schema import (
    LlamaModelParams,
    GenerationParams,
)
from ataraxai.app_logic.utils.config_schemas.whisper_config_schema import (
    WhisperModelParams,
)

# from ataraxai.app_logic.preferences_manager import PreferencesManager
# from ataraxai.app_logic.modules.rag.resilient_indexer import start_rag_file_monitoring
import os


def init_params():
    """
    Initialize and return the parameters for Llama and Whisper models.
    """
    project_dir = Path(__file__).resolve().parent.parent.parent

    llama_params = core_ai_py.LlamaModelParams.from_dict(
        LlamaModelParams(
            model_path=str(
                project_dir
                / "data/last_models/models/llama/Qwen3-30B-A3B-UD-IQ1_S.gguf"
            ),
            n_ctx=8192,
            n_gpu_layers=32,
            main_gpu=0,
            tensor_split=False,
            vocab_only=False,
            use_map=False,
            use_mlock=False,
        ).to_dict()
    )

    llama_generation_params = core_ai_py.GenerationParams.from_dict(
        GenerationParams(
            n_predict=128,
            temp=0.8,
            top_k=40,
            top_p=0.95,
            repeat_penalty=1.2,
            penalty_last_n=64,
            penalty_freq=0.7,
            penalty_present=0.0,
            stop_sequences=["</s>", "\n\n", "User:"],
            n_batch=1,
            n_threads=4,
        ).to_dict()
    )

    whisper_params = core_ai_py.WhisperModelParams.from_dict(
        WhisperModelParams(
            model=str(project_dir / "data/last_models/models/whisper/ggml-base.bin")
        ).to_dict()
    )

    return llama_params, llama_generation_params, whisper_params


def starter():
    """
    Initialize the AtaraxAI application.
    """

    llama_params, llama_generation_params, whisper_params = init_params()

    project_dir = Path(__file__).resolve().parent.parent.parent
    print(project_dir)

    core_ai_service = core_ai_py.CoreAIService()
    print(core_ai_service)
    print("AtaraxAI initialized successfully.")

    core_ai_service.initialize_llama_model(llama_params)
    core_ai_service.initialize_whisper_model(whisper_params)

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
