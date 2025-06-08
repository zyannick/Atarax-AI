from pydantic import BaseModel, Field
from typing import List


class SoundRecordingParams(BaseModel):
    sample_rate: int = Field(
        default=16000, description="Sample rate for audio recording"
    )
    frame_duration_ms: int = Field(
        default=30, description="Duration of each audio frame in milliseconds"
    )
    channels: int = Field(
        default=1, description="Number of audio channels (1 for mono, 2 for stereo)"
    )
    max_silence_ms: int = Field(
        default=800,
        description="Maximum silence duration in milliseconds before stopping the recording",
    )
    use_vad: bool = Field(
        default=True,
        description="Enable Voice Activity Detection (VAD) to filter out silence",
    )
    vad_mode: int = Field(
        default=2,
        description="VAD aggressiveness mode (0-3), higher means more aggressive filtering",
    )
    max_recording_duration: int = Field(
        default=60, description="Maximum recording duration in seconds"
    )
    format: str = Field(
        default="wav", description="Audio format for the recording (e.g., wav, mp3)"
    )
    
    def to_dict(self):
        return {
            "sample_rate": self.sample_rate,
            "frame_duration_ms": self.frame_duration_ms,
            "channels": self.channels,
            "max_silence_ms": self.max_silence_ms,
            "use_vad": self.use_vad,
            "vad_mode": self.vad_mode,
            "max_recording_duration": self.max_recording_duration,
            "format": self.format
        }
