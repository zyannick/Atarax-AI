import tempfile
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from pathlib import Path
from typing import List
import librosa
import soundfile as sf  # type: ignore
from ataraxai.praxis.modules.rag.parser.document_base_parser import (
    DocumentChunk,
    DocumentParser,
)
from ataraxai import hegemonikon_py  # type: ignore [attr-defined]
from typing import Dict, Any
from ataraxai.praxis.utils.config_schemas.whisper_config_schema import (
    WhisperModelParams,
    WhisperTranscriptionParams,
)


class AudioParser(DocumentParser):
    def __init__(
        self,
        whisper_transcribe: bool,
        core_ai_service: hegemonikon_py.CoreAIService,  # type: ignore
        transcription_params: hegemonikon_py.WhisperGenerationParams,  # type: ignore
        chunk_duration_seconds: int = 30,
        overlap_seconds: int = 5,
        max_file_size_mb: int = 100,
    ):
        """
        Initializes the audio parser with transcription and chunking parameters.

        Args:
            whisper_transcribe (bool): Whether to enable audio transcription using Whisper.
            core_ai_service (hegemonikon_py.CoreAIService): The core AI service instance for processing audio.
            transcription_params (hegemonikon_py.WhisperGenerationParams): Parameters for Whisper transcription.
            chunk_duration_seconds (int, optional): Duration (in seconds) of each audio chunk. Defaults to 30.
            overlap_seconds (int, optional): Overlap duration (in seconds) between audio chunks. Defaults to 5.
            max_file_size_mb (int, optional): Maximum allowed file size (in megabytes) for processing. Defaults to 100.
        """
        self.transcribe_audio = whisper_transcribe
        self.core_ai_service = core_ai_service  # type: ignore
        self.transcription_params = transcription_params  # type: ignore
        self.chunk_duration = chunk_duration_seconds
        self.overlap_duration = overlap_seconds
        self.max_file_size_mb = max_file_size_mb

    def should_use_chunking(self, audio_path: Path) -> bool:
        """
        Determines whether audio chunking should be used for a given audio file.

        Chunking is recommended if the file size exceeds the maximum allowed size or if the audio duration is longer than 5 minutes (300 seconds).
        If the duration cannot be determined, the decision is based solely on file size.

        Args:
            audio_path (Path): The path to the audio file.

        Returns:
            bool: True if chunking should be used, False otherwise.
        """
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)

        if file_size_mb > self.max_file_size_mb:
            return True

        try:
            duration: float = librosa.get_duration(path=str(audio_path))  # type: ignore
            return duration > 300
        except Exception:
            return file_size_mb > self.max_file_size_mb

    def create_audio_chunks(self, audio_path: Path) -> List[Path]:
        """
        Splits an audio file into overlapping chunks, resamples each chunk to 16kHz, and saves them as temporary WAV files.
        Args:
            audio_path (Path): Path to the input audio file.
        Returns:
            List[Path]: List of file paths to the generated audio chunks.
        Raises:
            Exception: If an error occurs during audio processing or chunk creation.
        Notes:
            - The chunk duration and overlap duration are determined by the instance attributes `self.chunk_duration` and `self.overlap_duration` (in seconds).
            - Each chunk is resampled to 16kHz before being saved.
            - Temporary files are created for each chunk and are not automatically deleted.
        """
        try:
            y, sr = librosa.load(str(audio_path), sr=None)  # type: ignore
            duration = len(y) / sr

            print(f"Audio duration: {duration:.2f} seconds, sample rate: {sr}")

            chunk_samples = int(self.chunk_duration * sr)
            overlap_samples = int(self.overlap_duration * sr)

            chunk_paths: List[Path] = []

            start = 0
            chunk_idx = 0

            while start < len(y):
                end = min(start + chunk_samples, len(y))
                chunk = y[start:end]  # type: ignore
                
                chunk_16k = librosa.resample(chunk, orig_sr=sr, target_sr=16000) # type: ignore

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav", prefix=f"chunk_{chunk_idx}_"
                )
                temp_path = Path(temp_file.name)
                temp_file.close()

                sf.write(str(temp_path), chunk_16k, 16000)  # type: ignore
                chunk_paths.append(temp_path)  # type: ignore

                if end >= len(y):
                    break
                start = end - overlap_samples
                chunk_idx += 1

            print(f"Created {len(chunk_paths)} chunks")
            return chunk_paths

        except Exception as e:
            print(f"Error creating chunks: {e}")
            raise

    def transcribe_chunks(self, chunk_paths: List[Path]) -> str:
        """
        Transcribes a list of audio file chunks and concatenates their transcriptions.

        Args:
            chunk_paths (List[Path]): List of file paths to audio chunks to be transcribed.

        Returns:
            str: The concatenated transcription of all audio chunks.

        Raises:
            ValueError: If all chunks fail to transcribe.

        Notes:
            - Each chunk is transcribed using the core AI service.
            - Overlapping text between consecutive chunks is removed.
            - Each chunk file is deleted after processing, regardless of success or failure.
            - If a chunk fails to transcribe, a warning is printed and processing continues with the next chunk.
        """
        transcriptions: List[str] = []

        for i, chunk_path in enumerate(chunk_paths):
            try:
                print(f"Transcribing chunk {i+1}/{len(chunk_paths)}")

                chunk_text = self.core_ai_service.transcribe_audio_file(  # type: ignore
                    str(chunk_path), self.transcription_params  # type: ignore
                )

                if chunk_text and chunk_text.strip():  # type: ignore
                    if i > 0 and transcriptions:
                        chunk_text = self.remove_overlap(transcriptions[-1], chunk_text)

                    transcriptions.append(str(chunk_text).strip())

            except Exception as e:
                print(f"Warning: Failed to transcribe chunk {i+1}: {e}")
                continue
            finally:
                try:
                    chunk_path.unlink()
                except Exception:
                    pass

        if not transcriptions:
            raise ValueError("All chunks failed to transcribe")

        return " ".join(transcriptions)

    def remove_overlap(self, previous_text: str, current_text: str) -> str:
        """
        Removes overlapping words between the end of `previous_text` and the start of `current_text`.

        This function compares the last words of `previous_text` with the first words of `current_text`
        (up to a maximum of 20 words) and removes the overlapping segment from `current_text` if found.

        Args:
            previous_text (str): The text segment preceding the current segment.
            current_text (str): The current text segment to process.

        Returns:
            str: The `current_text` with any overlapping prefix (matching the end of `previous_text`) removed.
        """

        previous_words = previous_text.split()
        current_words = current_text.split()

        max_overlap = min(len(previous_words), len(current_words), 20)

        for overlap_len in range(max_overlap, 0, -1):
            if previous_words[-overlap_len:] == current_words[:overlap_len]:
                return " ".join(current_words[overlap_len:])

        return current_text

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribes the given audio file to text.

        If the audio file is large, it is split into smaller chunks and each chunk is transcribed separately.
        Otherwise, the entire file is transcribed directly. Raises an error if transcription fails or returns no text.

        Args:
            audio_path (Path): The path to the audio file to be transcribed.

        Returns:
            str: The transcribed text from the audio file.

        Raises:
            ValueError: If transcription fails or returns no text.
            Exception: For any other errors encountered during transcription.
        """
        try:
            if self.should_use_chunking(audio_path):
                print(f"File {audio_path.name} is large, using chunked transcription")
                chunk_paths = self.create_audio_chunks(audio_path)
                audio_text = self.transcribe_chunks(chunk_paths)
            else:
                print(
                    f"File {audio_path.name} is small enough for direct transcription"
                )
                audio_text = self.core_ai_service.transcribe_audio_file(  # type: ignore
                    str(audio_path), self.transcription_params  # type: ignore
                )

            if not audio_text or not audio_text.strip():  # type: ignore
                raise ValueError(
                    f"Transcription failed for {audio_path}. No text returned."
                )

            return audio_text.strip()  # type: ignore

        except Exception as e:
            print(f"Transcription error for {audio_path}: {e}")
            raise

    def parse(self, path: Path) -> List[DocumentChunk]:
        """
        Parses an audio file to extract metadata and optionally transcribe audio to text.

        Args:
            path (Path): The path to the audio file to be parsed.

        Returns:
            List[DocumentChunk]: A list containing at least one DocumentChunk with extracted metadata.
                If audio transcription is enabled, a second DocumentChunk containing the transcribed text is included.
        """
        try:
            if str(path).lower().endswith(".mp3"):
                audio = MP3(str(path), ID3=EasyID3)
            elif str(path).lower().endswith(".wav"):
                audio = WAVE(str(path))  # type: ignore
            elif str(path).lower().endswith((".flac", ".ogg", ".m4a", ".aac", ".opus")):
                audio = EasyID3(str(path))  # type: ignore
            else: 
                raise ValueError(f"Unsupported audio format: {path.suffix}")

            tags: Dict[str, Any] = {}
            if hasattr(audio, "items") and audio.items():
                tags = {
                    k: v[0] if isinstance(v, list) and v else v
                    for k, v in audio.items()  # type: ignore
                }

            metadata_content = (
                "\n".join([f"{k.title()}: {v}" for k, v in tags.items()])
                if tags
                else "No metadata available"
            )

            base_chunk = DocumentChunk(
                content=metadata_content,
                source=str(path),
                metadata={
                    "type": "music",
                    "file_size_mb": path.stat().st_size / (1024 * 1024),
                    **tags,
                },
            )

            result_chunks = [base_chunk]

            # Add transcription if enabled
            if self.transcribe_audio:
                try:
                    transcription_text = self.transcribe(path)
                    transcription_chunk = DocumentChunk(
                        content=transcription_text,
                        source=str(path),
                        metadata={"type": "transcription", **tags},
                    )
                    result_chunks.append(transcription_chunk)
                except Exception as e:
                    print(f"Warning: Transcription failed for {path}: {e}")
                    error_chunk = DocumentChunk(
                        content=f"Transcription failed: {str(e)}",
                        source=str(path),
                        metadata={"type": "transcription_error", **tags},
                    )
                    result_chunks.append(error_chunk)

            return result_chunks

        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return [
                DocumentChunk(
                    content=f"Failed to parse audio file: {str(e)}",
                    source=str(path),
                    metadata={"type": "error", "error": str(e)},
                )
            ]


if __name__ == "__main__":
    whisper_params = hegemonikon_py.WhisperModelParams.from_dict(  # type: ignore
        WhisperModelParams(
            model=str("data/last_models/models/whisper/ggml-base.bin"),
            use_gpu=False,
            flash_attn=False,
            language= "en",
        ).model_dump( ) # type: ignore
    )

    whisper_generation_params = hegemonikon_py.WhisperGenerationParams.from_dict(  # type: ignore
        WhisperTranscriptionParams().model_dump()
    )

    core_ai_service = hegemonikon_py.CoreAIService()  # type: ignore
    core_ai_service.initialize_whisper_model(whisper_params)  # type: ignore

    parser = AudioParser(
        whisper_transcribe=True,
        core_ai_service=core_ai_service,  # type: ignore
        transcription_params=whisper_generation_params,
    )
    chunks = parser.parse(Path("tests/python/assets/test_audio.mp3"))
    for chunk in chunks:
        print(chunk)
