from pydantic import BaseModel, Field


class WhisperModelParams(BaseModel):
    n_threads: int = Field(default=0, description="Number of threads to use for processing.")
    use_gpu: bool = Field(default=True, description="Whether to use GPU for processing.")
    flash_attn: bool = Field(default=True, description="Whether to use flash attention.")
    audio_ctx: int = Field(default=0, description="Audio context size.")
    model: str = Field(default="data/last_models/models/whisper/ggml-base.en.bin", description="Path to the model file.")
    language: str = Field(default="en", description="Language for the model.")


class WhisperTranscriptionParams(BaseModel):
    config_version: str = Field(default="1.0", description="Version of the transcription configuration.")
    n_threads: int = Field(default=0, description="Number of threads to use for processing.")
    language: str = Field(default="en", description="Language for the transcription.")
    translate: bool = Field(default=False, description="Whether to enable translation.")
    print_special: bool = Field(default=False, description="Whether to print special tokens.")
    print_progress: bool = Field(default=True, description="Whether to print progress.")
    no_context: bool = Field(default=True, description="Whether to disable context.")
    max_len: int = Field(default=512, description="Maximum length of the input.")
    single_segment: bool = Field(default=False, description="Whether to use a single segment.")
    temperature: float = Field(default=0.8, description="Temperature for sampling.")


class WhisperConfig(BaseModel):
    config_version: str = Field(default="1.0", description="Version of the Whisper configuration.")
    whisper_model_params: WhisperModelParams = WhisperModelParams()
    whisper_transcription_params: WhisperTranscriptionParams = (
        WhisperTranscriptionParams()
    )
