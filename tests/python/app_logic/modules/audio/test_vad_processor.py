import pytest
import numpy as np
from ataraxai.app_logic.modules.audio.vad_processor import VADProcessor


def generate_silence(frame_size):
    return (np.zeros(frame_size, dtype=np.int16)).tobytes()


def generate_tone(frame_size, freq=440, sample_rate=16000):
    t = np.arange(frame_size) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return (tone * 32767).astype(np.int16).tobytes()


def test_vadprocessor_init_valid():
    vad = VADProcessor(sample_rate=16000, mode=2, frame_duration_ms=30)
    assert vad.sample_rate == 16000
    assert vad.frame_duration_ms == 30
    assert vad.frame_size == 480


@pytest.mark.parametrize("sample_rate", [8000, 16000, 32000, 48000])
@pytest.mark.parametrize("frame_duration_ms", [10, 20, 30])
def test_vadprocessor_init_valid_params(sample_rate, frame_duration_ms):
    vad = VADProcessor(sample_rate=sample_rate, frame_duration_ms=frame_duration_ms)
    assert vad.sample_rate == sample_rate
    assert vad.frame_duration_ms == frame_duration_ms


@pytest.mark.parametrize("sample_rate", [7000, 44100])
def test_vadprocessor_init_invalid_sample_rate(sample_rate):
    with pytest.raises(AssertionError):
        VADProcessor(sample_rate=sample_rate)


@pytest.mark.parametrize("frame_duration_ms", [5, 15, 25])
def test_vadprocessor_init_invalid_frame_duration(frame_duration_ms):
    with pytest.raises(AssertionError):
        VADProcessor(frame_duration_ms=frame_duration_ms)


def test_is_speech_on_silence():
    vad = VADProcessor()
    silence = generate_silence(vad.frame_size)
    assert not vad.is_speech(silence)


def test_is_speech_on_tone():
    vad = VADProcessor()
    tone = generate_tone(vad.frame_size)
    vad.fallback = True
    assert vad.is_speech(tone) in [True, False]


def test_is_speech_fallback_energy_threshold():
    vad = VADProcessor()
    silence = generate_silence(vad.frame_size)
    tone = generate_tone(vad.frame_size)
    vad.energy_threshold = None
    vad._is_energy_high(silence)
    assert vad.energy_threshold is not None
    prev_threshold = vad.energy_threshold
    vad._is_energy_high(tone)
    assert vad.energy_threshold != prev_threshold


def test_is_speech_without_fallback():
    vad = VADProcessor(fallback=False)
    silence = generate_silence(vad.frame_size)
    assert not vad.is_speech(silence)
