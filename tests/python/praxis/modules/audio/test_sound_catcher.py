import pytest
from unittest.mock import patch, MagicMock
from ataraxai.praxis.modules.audio.sound_catcher import SoundCatcher
import threading
import queue


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
def test_init_without_device_uses_default(
    mock_vad, mock_sd, mock_sound_recording_params
):
    mock_stream = MagicMock()
    mock_sd.RawInputStream.return_value = mock_stream
    mock_sd.default.device = [42, 43]

    catcher = SoundCatcher(sound_recording_params=mock_sound_recording_params)

    assert catcher.device == 42
    mock_sd.RawInputStream.assert_called_once()
    assert catcher.stream == mock_stream


@patch("ataraxai.app_logic.modules.audio.sound_catcher.sd")
@patch("ataraxai.app_logic.modules.audio.sound_catcher.VADProcessor")
def test_start_and_stop(mock_vad, mock_sd, mock_sound_recording_params):
    mock_stream = MagicMock()
    mock_sd.RawInputStream.return_value = mock_stream

    catcher = SoundCatcher(device=1, sound_recording_params=mock_sound_recording_params)

    with patch.object(threading.Thread, "start") as mock_thread_start:
        catcher.start()
        assert catcher.is_running is True
        mock_stream.start.assert_called_once()
        mock_thread_start.assert_called_once()
        assert catcher.processing_thread is not None

    with patch.object(catcher.processing_thread, "join") as mock_join:
        catcher.stop()
        assert catcher.is_running is False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_join.assert_called_once_with(timeout=1.0)


@patch("ataraxai.app_logic.modules.audio.sound_catcher.sd")
@patch("ataraxai.app_logic.modules.audio.sound_catcher.VADProcessor")
def test_callback_puts_data_in_queue(mock_vad, mock_sd, mock_sound_recording_params):
    catcher = SoundCatcher(device=2, sound_recording_params=mock_sound_recording_params)
    catcher.sound_queue = queue.Queue()
    indata = b"1234"
    frames = 2
    time = None
    status = None

    catcher.callback(indata, frames, time, status)
    assert not catcher.sound_queue.empty()
    assert catcher.sound_queue.get() == indata


@patch("ataraxai.app_logic.modules.audio.sound_catcher.sd")
@patch("ataraxai.app_logic.modules.audio.sound_catcher.VADProcessor")
def test_callback_prints_status(mock_vad, mock_sd, mock_sound_recording_params, capsys):
    catcher = SoundCatcher(device=2, sound_recording_params=mock_sound_recording_params)
    status = "SomeStatus"
    catcher.callback(b"data", 1, None, status)
    captured = capsys.readouterr()
    assert "Audio callback status: SomeStatus" in captured.out


@patch("ataraxai.app_logic.modules.audio.sound_catcher.sd")
@patch("ataraxai.app_logic.modules.audio.sound_catcher.VADProcessor")
def test_process_audio_segment_prints_info(
    mock_vad, mock_sd, mock_sound_recording_params, capsys
):
    catcher = SoundCatcher(device=2, sound_recording_params=mock_sound_recording_params)
    audio_bytes = b"\x01\x00" * 10  
    catcher.process_audio_segment(audio_bytes)
    captured = capsys.readouterr()
    assert "Processing audio segment: 10 samples" in captured.out


@patch("ataraxai.app_logic.modules.audio.sound_catcher.sd")
@patch("ataraxai.app_logic.modules.audio.sound_catcher.VADProcessor")
def test_catch_sound_handles_queue_empty_and_exception(
    mock_vad, mock_sd, mock_sound_recording_params, capsys
):
    catcher = SoundCatcher(device=2, sound_recording_params=mock_sound_recording_params)
    catcher.is_running = True

    catcher.sound_queue.get = MagicMock(
        side_effect=[queue.Empty(), Exception("Test error")]
    )
    with patch.object(catcher, "process_audio_segment"):
        catcher.catch_sound()
    captured = capsys.readouterr()
    assert "Error in catch_sound: Test error" in captured.out


@patch("ataraxai.app_logic.modules.audio.sound_catcher.sd")
@patch("ataraxai.app_logic.modules.audio.sound_catcher.VADProcessor")
def test_process_audio_segment_prints_correct_sample_count(
    mock_vad, mock_sd, mock_sound_recording_params, capsys
):
    catcher = SoundCatcher(device=2, sound_recording_params=mock_sound_recording_params)
    audio_bytes = b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00"
    catcher.process_audio_segment(audio_bytes)
    captured = capsys.readouterr()
    assert "Processing audio segment: 5 samples" in captured.out


@patch("ataraxai.app_logic.modules.audio.sound_catcher.sd")
@patch("ataraxai.app_logic.modules.audio.sound_catcher.VADProcessor")
def test_process_audio_segment_empty_bytes(
    mock_vad, mock_sd, mock_sound_recording_params, capsys
):
    catcher = SoundCatcher(device=2, sound_recording_params=mock_sound_recording_params)
    audio_bytes = b""
    catcher.process_audio_segment(audio_bytes)
    captured = capsys.readouterr()
    assert "Processing audio segment: 0 samples" in captured.out
