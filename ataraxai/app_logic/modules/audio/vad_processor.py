import webrtcvad
import numpy as np


class VADProcessor:
    def __init__(self, sample_rate=16000, mode=2, frame_duration_ms=30, fallback=True):
        """
        :param sample_rate: Hz, only 8000, 16000, 32000, or 48000 are valid for webrtcvad.
        :param mode: Aggressiveness mode (0-3). Higher = more filtering (less false positives).
        :param frame_duration_ms: 10, 20, or 30ms.
        :param fallback: If True, enables energy-based fallback detection.
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
        try:
            if self.vad.is_speech(frame_bytes, self.sample_rate):
                return True
        except Exception:
            return False

        if self.fallback:
            return self._is_energy_high(frame_bytes)

        return False

    def _is_energy_high(self, frame_bytes):
        frame_np = np.frombuffer(frame_bytes, dtype=np.int16)
        rms_energy = np.sqrt(np.mean(np.square(frame_np)))

        if self.energy_threshold is None:
            self.energy_threshold = rms_energy * 1.5
        else:
            self.energy_threshold = 0.9 * self.energy_threshold + 0.1 * rms_energy * 1.5

        return rms_energy > self.energy_threshold
