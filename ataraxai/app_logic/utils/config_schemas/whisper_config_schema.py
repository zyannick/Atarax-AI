from pydantic import BaseModel


class WhisperModelParams(BaseModel):
    n_threads: int = 0
    use_gpu: bool = True
    flash_attn: bool = True
    audio_ctx: int = 0
    model: str = "models/ggml-base.en.bin"
    language: str = "en"


class WhisperTranscriptionParams(BaseModel):
    config_version: float = 1.0
    n_threads: int = 0
    language: str = "en"
    translate: bool = False
    print_special: bool = False
    print_progress: bool = True
    no_context: bool = True
    max_len: int = 512
    single_segment: bool = False
    temperature: float = 0.8


class WhisperConfig(BaseModel):
    config_version: float = 1.0
    whisper_model_params: WhisperModelParams = WhisperModelParams()
    whisper_transcription_params: WhisperTranscriptionParams = (
        WhisperTranscriptionParams()
    )
