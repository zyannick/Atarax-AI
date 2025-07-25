from pydantic import BaseModel, Field


class WhisperModelParams(BaseModel):
    n_threads: int = Field(0, description="Number of threads to use for processing.")
    use_gpu: bool = Field(True, description="Whether to use GPU for processing.")
    flash_attn: bool = Field(True, description="Whether to use flash attention.")
    audio_ctx: int = Field(0, description="Audio context size.")
    model: str = Field("data/last_models/models/whisper/ggml-base.en.bin", description="Path to the model file.")
    language: str = Field("en", description="Language for the model.")


class WhisperTranscriptionParams(BaseModel):
    config_version: float = Field(1.0, description="Version of the transcription configuration.")
    n_threads: int = Field(0, description="Number of threads to use for processing.")
    language: str = Field("en", description="Language for the transcription.")
    translate: bool = Field(False, description="Whether to enable translation.")
    print_special: bool = Field(False, description="Whether to print special tokens.")
    print_progress: bool = Field(True, description="Whether to print progress.")
    no_context: bool = Field(True, description="Whether to disable context.")
    max_len: int = Field(512, description="Maximum length of the input.")
    single_segment: bool = Field(False, description="Whether to use a single segment.")
    temperature: float = Field(0.8, description="Temperature for sampling.")


class WhisperConfig(BaseModel):
    config_version: float = Field(1.0, description="Version of the Whisper configuration.")
    whisper_model_params: WhisperModelParams = WhisperModelParams()
    whisper_transcription_params: WhisperTranscriptionParams = (
        WhisperTranscriptionParams()
    )
