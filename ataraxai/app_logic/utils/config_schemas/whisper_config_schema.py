from pydantic import BaseModel, Field
from typing import List


class WhisperModelParams(BaseModel):
    model_name: str = "base"
    use_gpu: bool = True


class WhisperTranscriptionParams(BaseModel):
    n_threads: int = 0
    language: str = "en"
    translate: bool = False
    print_special: bool = False
    print_progress: bool = True
    no_context: bool = True
    max_len: int = 0
    single_segment: bool = False
    temperature: float = 0.0


class WhisperConfig(BaseModel):
    config_version: float = 1.0
    whisper_model_params: WhisperModelParams = WhisperModelParams()
    whisper_transcription_params: WhisperTranscriptionParams = (
        WhisperTranscriptionParams()
    )
