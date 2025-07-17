import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.modules.rag.parser.audio_parser import AudioParser
from ataraxai.praxis.modules.rag.parser.document_base_parser import DocumentChunk
import numpy as np

@pytest.fixture
def mock_core_ai_service():
    return mock.Mock()

@pytest.fixture
def dummy_transcription_params():
    return mock.Mock()

@pytest.fixture
def parser(mock_core_ai_service, dummy_transcription_params):
    return AudioParser(
        whisper_transcribe=True,
        core_ai_service=mock_core_ai_service,
        transcription_params=dummy_transcription_params,
    )

@pytest.fixture
def fake_mp3_path(tmp_path):
    file = tmp_path / "test.mp3"
    file.write_bytes(b"ID3")  # minimal MP3 header
    return file

@pytest.fixture
def fake_wav_path(tmp_path):
    file = tmp_path / "test.wav"
    file.write_bytes(b"RIFF")  # minimal WAV header
    return file

@pytest.fixture
def fake_flac_path(tmp_path):
    file = tmp_path / "test.flac"
    file.write_bytes(b"fLaC")  # minimal FLAC header
    return file

def test_parse_mp3_metadata_only(parser, fake_mp3_path):
    with mock.patch("ataraxai.praxis.modules.rag.parser.audio_parser.MP3") as mock_mp3:
        mock_audio = mock.Mock()
        mock_audio.items.return_value = [("artist", ["Test Artist"]), ("album", ["Test Album"])]
        mock_mp3.return_value = mock_audio
        parser.transcribe_audio = False

        chunks = parser.parse(fake_mp3_path)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, DocumentChunk)
        assert "Artist: Test Artist" in chunk.content
        assert "Album: Test Album" in chunk.content
        assert chunk.metadata["type"] == "music"
        assert "artist" in chunk.metadata
        assert "album" in chunk.metadata

def test_parse_wav_metadata_only(parser, fake_wav_path):
    with mock.patch("ataraxai.praxis.modules.rag.parser.audio_parser.WAVE") as mock_wave:
        mock_audio = mock.Mock()
        mock_audio.items.return_value = []
        mock_wave.return_value = mock_audio
        parser.transcribe_audio = False

        chunks = parser.parse(fake_wav_path)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "No metadata available" in chunk.content
        assert chunk.metadata["type"] == "music"

def test_parse_flac_metadata_only(parser, fake_flac_path):
    with mock.patch("ataraxai.praxis.modules.rag.parser.audio_parser.EasyID3") as mock_easyid3:
        mock_audio = mock.Mock()
        mock_audio.items.return_value = [("title", ["Test Title"])]
        mock_easyid3.return_value = mock_audio
        parser.transcribe_audio = False

        chunks = parser.parse(fake_flac_path)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "Title: Test Title" in chunk.content
        assert chunk.metadata["type"] == "music"

def test_parse_unsupported_format(parser, tmp_path):
    file = tmp_path / "test.unsupported"
    file.write_bytes(b"data")
    chunks = parser.parse(file)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "Failed to parse audio file" in chunk.content
    assert chunk.metadata["type"] == "error"

def test_parse_with_transcription_success(parser, fake_mp3_path, mock_core_ai_service):
    with mock.patch("ataraxai.praxis.modules.rag.parser.audio_parser.MP3") as mock_mp3:
        mock_audio = mock.Mock()
        mock_audio.items.return_value = [("artist", ["Test Artist"])]
        mock_mp3.return_value = mock_audio
        parser.transcribe_audio = True
        parser.transcribe = mock.Mock(return_value="Transcribed text")

        chunks = parser.parse(fake_mp3_path)
        assert len(chunks) == 2
        assert chunks[1].content == "Transcribed text"
        assert chunks[1].metadata["type"] == "transcription"

def test_parse_with_transcription_failure(parser, fake_mp3_path):
    with mock.patch("ataraxai.praxis.modules.rag.parser.audio_parser.MP3") as mock_mp3:
        mock_audio = mock.Mock()
        mock_audio.items.return_value = []
        mock_mp3.return_value = mock_audio
        parser.transcribe_audio = True
        parser.transcribe = mock.Mock(side_effect=Exception("transcription failed"))

        chunks = parser.parse(fake_mp3_path)
        assert len(chunks) == 2
        assert "Transcription failed" in chunks[1].content
        assert chunks[1].metadata["type"] == "transcription_error"
        
        
def test_should_use_chunking_large_file(parser, fake_mp3_path):
    # Simulate a large file by patching stat().st_size
    with mock.patch.object(Path, "stat") as mock_stat, \
            mock.patch("librosa.get_duration", return_value=100):
        mock_stat.return_value.st_size = 200 * 1024 * 1024  # 200 MB
        assert parser.should_use_chunking(fake_mp3_path) is True

def test_should_use_chunking_long_duration(parser, fake_mp3_path):
    # Simulate a small file but long duration
    with mock.patch.object(Path, "stat") as mock_stat, \
            mock.patch("librosa.get_duration", return_value=400):
        mock_stat.return_value.st_size = 1 * 1024 * 1024  # 1 MB
        assert parser.should_use_chunking(fake_mp3_path) is True

def test_should_use_chunking_short_file(parser, fake_mp3_path):
    # Simulate a small file and short duration
    with mock.patch.object(Path, "stat") as mock_stat, \
            mock.patch("librosa.get_duration", return_value=100):
        mock_stat.return_value.st_size = 1 * 1024 * 1024  # 1 MB
        assert parser.should_use_chunking(fake_mp3_path) is False

def test_create_audio_chunks_creates_files(parser, tmp_path):
    fake_audio_path = tmp_path / "audio.wav"
    # Patch librosa.load and librosa.resample, and sf.write
    y = np.random.randn(32000)  # 2 seconds at 16kHz
    sr = 16000
    with mock.patch("librosa.load", return_value=(y, sr)), \
            mock.patch("librosa.resample", side_effect=lambda x, orig_sr, target_sr: x), \
            mock.patch("soundfile.write") as mock_sf_write:
        parser.chunk_duration = 1  # 1 second
        parser.overlap_duration = 0
        chunk_paths = parser.create_audio_chunks(fake_audio_path)
        assert len(chunk_paths) == 2
        for path in chunk_paths:
            assert path.suffix == ".wav"
        assert mock_sf_write.call_count == 2

def test_transcribe_chunks_success(parser, tmp_path):
    # Prepare fake chunk files
    chunk1 = tmp_path / "chunk1.wav"
    chunk2 = tmp_path / "chunk2.wav"
    chunk1.write_bytes(b"data")
    chunk2.write_bytes(b"data")
    parser.core_ai_service.transcribe_audio_file.side_effect = ["hello world", "world again"]
    # Patch remove_overlap to just concatenate
    parser.remove_overlap = lambda prev, curr: curr
    result = parser.transcribe_chunks([chunk1, chunk2])
    assert "hello world" in result and "world again" in result

def test_transcribe_chunks_all_fail(parser, tmp_path):
    chunk1 = tmp_path / "chunk1.wav"
    chunk1.write_bytes(b"data")
    parser.core_ai_service.transcribe_audio_file.side_effect = Exception("fail")
    with pytest.raises(ValueError):
        parser.transcribe_chunks([chunk1])

def test_remove_overlap_removes_overlap(parser):
    prev = "hello world this is a test"
    curr = "this is a test and more"
    result = parser.remove_overlap(prev, curr)
    assert result == "and more"

def test_remove_overlap_no_overlap(parser):
    prev = "hello world"
    curr = "foo bar"
    result = parser.remove_overlap(prev, curr)
    assert result == "foo bar"

def test_transcribe_should_use_chunking(parser, fake_mp3_path):
    with mock.patch.object(parser, "should_use_chunking", return_value=True), \
            mock.patch.object(parser, "create_audio_chunks", return_value=[Path("chunk1.wav")]), \
            mock.patch.object(parser, "transcribe_chunks", return_value="chunked text"):
        result = parser.transcribe(fake_mp3_path)
        assert result == "chunked text"

def test_transcribe_direct(parser, fake_mp3_path):
    with mock.patch.object(parser, "should_use_chunking", return_value=False):
        parser.core_ai_service.transcribe_audio_file.return_value = "direct text"
        result = parser.transcribe(fake_mp3_path)
        assert result == "direct text"

def test_transcribe_raises_on_empty(parser, fake_mp3_path):
    with mock.patch.object(parser, "should_use_chunking", return_value=False):
        parser.core_ai_service.transcribe_audio_file.return_value = ""
        with pytest.raises(ValueError):
            parser.transcribe(fake_mp3_path)
