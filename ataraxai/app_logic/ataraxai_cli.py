from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from ataraxai import core_ai_py # type: ignore [attr-defined]
from ataraxai.app_logic.prompt_utils import create_prompt
from ataraxai.app_logic.utils.config_schemas.llama_config_schema import (
    GenerationParams, LlamaModelParams)
from ataraxai.app_logic.utils.config_schemas.whisper_config_schema import (
    WhisperModelParams)

PROMPTS_PROCESSED = Counter('ataraxai_prompts_processed_total', 'Total number of prompts processed')
ERRORS_TOTAL = Counter('ataraxai_errors_total', 'Total number of errors encountered', ['error_type'])
CONVERSATION_HISTORY_LENGTH = Gauge('ataraxai_conversation_history_length', 'The current length of the conversation history')
PROMPT_PROCESSING_DURATION = Histogram('ataraxai_prompt_processing_duration_seconds', 'Latency of prompt processing')
MODEL_INFO = Gauge('ataraxai_model_info', 'Information about the loaded models', ['model_type', 'model_path'])


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
    try:
        start_http_server(8000)
        print("Prometheus metrics server started on http://localhost:8000")
    except OSError as e:
        print(f"Prometheus server already running or port 8000 is in use: {e}")


    llama_params, llama_generation_params, whisper_params = init_params()

    project_dir = Path(__file__).resolve().parent.parent.parent
    print(project_dir)

    core_ai_service = core_ai_py.CoreAIService()
    print(core_ai_service)
    print("AtaraxAI initialized successfully.")

    core_ai_service.initialize_llama_model(llama_params)
    core_ai_service.initialize_whisper_model(whisper_params)

    MODEL_INFO.labels(model_type='llama', model_path=llama_params.model_path).set(1)
    MODEL_INFO.labels(model_type='whisper', model_path=whisper_params.model).set(1)

    conversation_history = []
    CONVERSATION_HISTORY_LENGTH.set(0) 

    while True:
        default_system_message = (
            "You are a helpful AI assistant. Provide clear and concise answers."
        )

        user_input_text = input("Enter your query (or type 'exit' to quit): ")
        if user_input_text.lower() == "exit":
            print("Exiting AtaraxAI.")
            break
        
        conversation_history.append(f"User: {user_input_text.strip()}")

        full_prompt = create_prompt(
            user_query=user_input_text, system_message=default_system_message, conversation_history=conversation_history
        )
        current_generation_params = llama_generation_params

        with PROMPT_PROCESSING_DURATION.time(): 
            answer = core_ai_service.process_prompt(full_prompt, current_generation_params)
        
        PROMPTS_PROCESSED.inc() 

        print(f"AtaraxAI assistant: {answer}")
        conversation_history.append(f"Assistant: {answer.strip()}")
        CONVERSATION_HISTORY_LENGTH.set(len(conversation_history)) 

        if len(conversation_history) > 10: 
            conversation_history = conversation_history[:1] + conversation_history[3:] 
            CONVERSATION_HISTORY_LENGTH.set(len(conversation_history))


if __name__ == "__main__":
    try:
        starter()
    except KeyboardInterrupt:
        print("\nAtaraxAI shutdown by user.")
    except Exception as e:
        ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        print(f"Error starting AtaraxAI: {e}")
        raise