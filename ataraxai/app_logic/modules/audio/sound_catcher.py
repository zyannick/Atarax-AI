import sounddevice as sd
from vad_processor import VADProcessor
import queue
import numpy as np
import datetime
import threading
from ataraxai.app_logic.utils.config_schemas.sound_recording_schema import (
    SoundRecordingParams,
)


class SoundCatcher:

    def __init__(
        self,
        device=None,
        sound_recording_params: SoundRecordingParams = SoundRecordingParams(),
    ):
        self.sound_recording_params = sound_recording_params
        self.device = device if device is not None else sd.default.device[0]
        self.sound_queue = queue.Queue()
        self.vad_processor = VADProcessor(
            sample_rate=sound_recording_params.sample_rate,
            mode=sound_recording_params.vad_mode,
            frame_duration_ms=sound_recording_params.frame_duration_ms,
            fallback=True,
        )

        self.stream = sd.RawInputStream(
            samplerate=sound_recording_params.sample_rate,
            blocksize=int(
                sound_recording_params.sample_rate
                * sound_recording_params.frame_duration_ms
                / 1000
            ),
            device=device,
            dtype="int16",
            channels=sound_recording_params.channels,
            callback=self.callback,
        )

        self.is_running = False
        self.processing_thread = None

    def callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")

        self.sound_queue.put(bytes(indata))

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.stream.start()
            self.processing_thread = threading.Thread(
                target=self.catch_sound, daemon=True
            )
            self.processing_thread.start()
            print("SoundCatcher started")

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.stream.stop()
            self.stream.close()
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
            print("SoundCatcher stopped")

    def catch_sound(self):
        speech_buffer = []
        last_speech_time = datetime.datetime.now()

        print("Listening for speech...")

        while self.is_running:
            try:
                frame = self.sound_queue.get(timeout=0.1)
                is_speech = self.vad_processor.is_speech(frame)

                if is_speech:
                    speech_buffer.append(frame)
                    last_speech_time = datetime.datetime.now()
                    if len(speech_buffer) == 1:
                        print("Speech detected, recording...")
                else:
                    silence_duration = (
                        datetime.datetime.now() - last_speech_time
                    ).total_seconds()

                    if (
                        silence_duration
                        > self.sound_recording_params.max_silence_ms / 1000
                        and speech_buffer
                    ):
                        # End of speech detected
                        audio_bytes = b"".join(speech_buffer)
                        duration_ms = (
                            len(speech_buffer)
                            * self.sound_recording_params.frame_duration_ms
                        )

                        print(
                            f"Speech segment captured ({duration_ms}ms, {len(audio_bytes)} bytes)"
                        )
                        self.process_audio_segment(audio_bytes)
                        speech_buffer = []

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in catch_sound: {e}")
                break

    def process_audio_segment(self, audio_bytes):
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        print(f"Processing audio segment: {len(audio_array)} samples")

        # whisper_transcribe(audio_bytes)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self):
        self.stop()
