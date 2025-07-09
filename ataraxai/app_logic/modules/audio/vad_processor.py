import webrtcvad
import numpy as np


class VADProcessor:
    def __init__(self, sample_rate=16000, mode=2, frame_duration_ms=30, fallback=True):
        """
        Initializes the VAD (Voice Activity Detection) processor.

        Args:
            sample_rate (int, optional): The audio sample rate in Hz. Must be one of [8000, 16000, 32000, 48000]. Defaults to 16000.
            mode (int, optional): The aggressiveness mode of the VAD. Higher values are more aggressive in filtering out non-speech. Defaults to 2.
            frame_duration_ms (int, optional): The frame duration in milliseconds. Must be one of [10, 20, 30]. Defaults to 30.
            fallback (bool, optional): Whether to enable fallback mechanism if VAD fails. Defaults to True.

        Raises:
            AssertionError: If sample_rate or frame_duration_ms are not in the allowed values.
        """
        assert sample_rate in [8000, 16000, 32000, 48000]
        assert frame_duration_ms in [10, 20, 30]

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) 
        self.vad = webrtcvad.Vad(mode)
        self.fallback = fallback
        self.energy_threshold = None

    def is_speech(self, frame_bytes):
        """
        Determines whether the given audio frame contains speech.

        Args:
            frame_bytes (bytes): The audio frame data in bytes.

        Returns:
            bool: True if speech is detected in the frame, False otherwise.

        Notes:
            - Uses the primary VAD (Voice Activity Detection) engine to detect speech.
            - If the primary VAD raises an exception and a fallback is enabled, uses an energy-based method as a fallback.
        """
        try:
            if self.vad.is_speech(frame_bytes, self.sample_rate):
                return True
        except Exception:
            return False

        if self.fallback:
            return self._is_energy_high(frame_bytes)

        return False

    def _is_energy_high(self, frame_bytes):
        """
        Determines whether the root mean square (RMS) energy of an audio frame exceeds a dynamic threshold.

        The threshold is initialized based on the first frame's energy and is then updated adaptively
        using a weighted average of the previous threshold and the current frame's energy.

        Args:
            frame_bytes (bytes): The audio frame data in bytes, expected to be 16-bit PCM.

        Returns:
            bool: True if the RMS energy of the frame is higher than the current energy threshold, False otherwise.
        """
        frame_np = np.frombuffer(frame_bytes, dtype=np.int16)
        rms_energy = np.sqrt(np.mean(np.square(frame_np)))

        if self.energy_threshold is None:
            self.energy_threshold = rms_energy * 1.5
        else:
            self.energy_threshold = 0.9 * self.energy_threshold + 0.1 * rms_energy * 1.5

        return rms_energy > self.energy_threshold
