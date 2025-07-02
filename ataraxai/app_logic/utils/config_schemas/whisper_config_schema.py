from pydantic import BaseModel


class WhisperModelParams(BaseModel):
    n_threads: int = 0
    use_gpu: bool = True
    flash_attn: bool = True
    audio_ctx : int = 0
    model: str = "models/ggml-base.en.bin"
    language: str = "en"
    
    def to_dict(self):
        return {
            "n_threads": self.n_threads,
            "use_gpu": self.use_gpu,
            "flash_attn": self.flash_attn,
            "audio_ctx": self.audio_ctx,
            "model": self.model,
            "language": self.language
        }


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
    
    def to_dict(self):
        return {
            "config_version": self.config_version,  
            "n_threads": self.n_threads,
            "language": self.language,
            "translate": self.translate,
            "print_special": self.print_special,
            "print_progress": self.print_progress,
            "no_context": self.no_context,
            "max_len": self.max_len,
            "single_segment": self.single_segment,
            "temperature": self.temperature
        }


class WhisperConfig(BaseModel):
    config_version: float = 1.0
    whisper_model_params: WhisperModelParams = WhisperModelParams()
    whisper_transcription_params: WhisperTranscriptionParams = (
        WhisperTranscriptionParams()
    )
    
    def to_dict(self):
        return {
            "config_version": self.config_version,
            "whisper_model_params": self.whisper_model_params.to_dict(),
            "whisper_transcription_params": self.whisper_transcription_params.to_dict()
        }
