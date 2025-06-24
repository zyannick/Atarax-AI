import pytest
from unittest.mock import patch, MagicMock
from ataraxai.app_logic.modules.audio.sound_catcher import SoundCatcher

@pytest.fixture
def mock_sound_recording_params():
    mock = MagicMock()
    mock.sample_rate = 16000
    mock.vad_mode = 1
    mock.frame_duration_ms = 30
    mock.channels = 1
    mock.max_silence_ms = 500
    return mock

@patch("ataraxai.app_logic.modules.audio.sound_catcher.sd")
@patch("ataraxai.app_logic.modules.audio.sound_catcher.VADProcessor")
def test_init_with_device(mock_vad, mock_sd, mock_sound_recording_params):
    mock_stream = MagicMock()
    mock_sd.RawInputStream.return_value = mock_stream

    catcher = SoundCatcher(device=3, sound_recording_params=mock_sound_recording_params)

    assert catcher.device == 3
    assert catcher.sound_recording_params == mock_sound_recording_params
    mock_vad.assert_called_once_with(
        sample_rate=mock_sound_recording_params.sample_rate,
        mode=mock_sound_recording_params.vad_mode,
        frame_duration_ms=mock_sound_recording_params.frame_duration_ms,
        fallback=True,
    )
    mock_sd.RawInputStream.assert_called_once()
    assert catcher.stream == mock_stream
    assert catcher.is_running is False
    assert catcher.processing_thread is None

@patch("ataraxai.app_logic.modules.audio.sound_catcher.sd")
@patch("ataraxai.app_logic.modules.audio.sound_catcher.VADProcessor")
def test_init_without_device_uses_default(mock_vad, mock_sd, mock_sound_recording_params):
    mock_stream = MagicMock()
    mock_sd.RawInputStream.return_value = mock_stream
    mock_sd.default.device = [42, 43]

    catcher = SoundCatcher(sound_recording_params=mock_sound_recording_params)

    assert catcher.device == 42
    mock_sd.RawInputStream.assert_called_once()
    assert catcher.stream == mock_stream