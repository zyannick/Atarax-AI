import pytest
from ataraxai.praxis.utils.configs.config_schemas.sound_recording_schema import SoundRecordingParams

def test_default_values():
    params = SoundRecordingParams()
    assert params.config_version == "1.0"
    assert params.sample_rate == 16000
    assert params.frame_duration_ms == 30
    assert params.channels == 1
    assert params.max_silence_ms == 800
    assert params.use_vad is True
    assert params.vad_mode == 2
    assert params.max_recording_duration == 60
    assert params.format == "wav"

def test_custom_values():
    params = SoundRecordingParams(
        config_version="2.0",
        sample_rate=44100,
        frame_duration_ms=50,
        channels=2,
        max_silence_ms=1000,
        use_vad=False,
        vad_mode=1,
        max_recording_duration=120,
        format="mp3"
    )
    assert params.config_version == "2.0"
    assert params.sample_rate == 44100
    assert params.frame_duration_ms == 50
    assert params.channels == 2
    assert params.max_silence_ms == 1000
    assert params.use_vad is False
    assert params.vad_mode == 1
    assert params.max_recording_duration == 120
    assert params.format == "mp3"

@pytest.mark.parametrize("invalid_sample_rate", [0, -16000, "not_an_int"])
def test_invalid_sample_rate(invalid_sample_rate):
    with pytest.raises(ValueError):
        SoundRecordingParams(sample_rate=invalid_sample_rate)

@pytest.mark.parametrize("vad_mode", [-1, 4, 100])
def test_invalid_vad_mode(vad_mode):
    params = SoundRecordingParams(vad_mode=vad_mode)
    # The schema does not restrict vad_mode, so it should accept any int
    assert params.vad_mode == vad_mode

def test_format_accepts_any_string():
    params = SoundRecordingParams(format="ogg")
    assert params.format == "ogg"