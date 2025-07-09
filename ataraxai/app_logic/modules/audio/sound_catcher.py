import sounddevice as sd
from ataraxai.app_logic.modules.audio.vad_processor import VADProcessor
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
        """
        Initializes the sound catcher object with the specified audio device and recording parameters.

        Args:
            device (Optional[int]): The audio input device index to use. If None, the default device is used.
            sound_recording_params (SoundRecordingParams): Configuration parameters for sound recording, including sample rate, VAD mode, frame duration, and number of channels.

        Attributes:
            sound_recording_params (SoundRecordingParams): Stores the recording parameters.
            device (int): The audio input device index.
            sound_queue (queue.Queue): Queue to store audio data chunks.
            vad_processor (VADProcessor): Voice Activity Detection processor initialized with the given parameters.
            stream (sd.RawInputStream): The audio input stream for capturing raw audio data.
            is_running (bool): Indicates whether the audio capture is currently running.
            processing_thread (Optional[threading.Thread]): Thread for processing audio data.
        """
        self.sound_recording_params = sound_recording_params
        self.device = device if device is not None else sd.default.device[0]
        self.sound_queue: queue.Queue = queue.Queue()
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
        """
        Callback function for audio input stream.

        Parameters:
            indata (numpy.ndarray): The recorded audio data as a NumPy array.
            frames (int): The number of frames in this block of audio data.
            time (CData): A CData structure containing timing information.
            status (CallbackFlags): Indicates whether any errors or warnings occurred.

        If a status is present, it prints the status message. The audio data is then
        converted to bytes and placed into the sound_queue for further processing.
        """
        if status:
            print(f"Audio callback status: {status}")

        self.sound_queue.put(bytes(indata))

    def start(self):
        """
        Starts the sound catching process if it is not already running.

        This method sets the `is_running` flag to True, starts the audio stream,
        and launches a background thread to process incoming sound data using the
        `catch_sound` method. If the sound catcher is already running, this method
        does nothing.

        Prints a message to indicate that the SoundCatcher has started.
        """
        if not self.is_running:
            self.is_running = True
            self.stream.start()
            self.processing_thread = threading.Thread(
                target=self.catch_sound, daemon=True
            )
            self.processing_thread.start()
            print("SoundCatcher started")

    def stop(self):
        """
        Stops the sound capturing process if it is currently running.

        This method sets the running flag to False, stops and closes the audio stream,
        and waits for the processing thread to finish (with a timeout). Prints a message
        when the sound catcher has stopped.
        """
        if self.is_running:
            self.is_running = False
            self.stream.stop()
            self.stream.close()
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
            print("SoundCatcher stopped")

    def catch_sound(self):
        """
        Continuously listens for audio frames, detects speech segments using a VAD processor,
        and processes captured speech segments when a period of silence is detected.

        The method maintains a buffer of audio frames while speech is detected. When silence
        exceeds the configured maximum duration, the buffered speech is considered a complete
        segment and is processed. Handles queue timeouts and logs errors.

        Returns:
            None
        """
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
        """
        Processes a segment of audio data provided as bytes.

        Converts the input audio bytes into a NumPy array of 16-bit integers,
        logs the number of samples, and prepares the data for further processing
        such as transcription.

        Args:
            audio_bytes (bytes): The raw audio data in bytes format.

        Returns:
            None
        """
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        print(f"Processing audio segment: {len(audio_array)} samples")

        # whisper_transcribe(audio_bytes)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self):
        self.stop()
