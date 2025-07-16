import subprocess
from pathlib import Path
from typing import List
from ataraxai.praxis.modules.rag.parser.document_base_parser import DocumentChunk, DocumentParser
import tempfile


class VideoParser(DocumentParser):
    def __init__(self, whisper_transcribe, core_ai_service, transcription_params):
        """
        Initializes the VideoParser with the given transcription and AI service components.

        Args:
            whisper_transcribe: The transcription engine or function used for processing audio/video files.
            core_ai_service: The core AI service instance used for further processing or analysis.
            transcription_params: Parameters or configuration settings for the transcription process.
        """
        self.core_ai_service = core_ai_service
        self.transcription_params = transcription_params
        self.whisper_transcribe = whisper_transcribe

    def extract_audio(self, video_path: Path, out_wav: Path):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribes the audio file at the given path into text.

        Args:
            audio_path (Path): The path to the audio file to be transcribed.

        Returns:
            str: The transcribed text from the audio file.

        Raises:
            ValueError: If transcription fails and no text is returned.
        """
        audio_text: str = self.core_ai_service.transcribe(
            audio_path, self.transcription_params
        )
        if not audio_text:
            raise ValueError(
                f"Transcription failed for {audio_path}. No text returned."
            )
        return audio_text

    def parse(self, path: Path) -> List[DocumentChunk]:
        """
        Parses a video file and returns a list of DocumentChunk objects containing metadata and optional transcript.

        Args:
            path (Path): The path to the video file to be parsed.

        Returns:
            List[DocumentChunk]: A list containing at least one DocumentChunk with video metadata. 
                If whisper_transcribe is enabled, includes an additional chunk with the transcribed audio content.

        Notes:
            - If whisper_transcribe is False, only metadata is returned.
            - Audio is temporarily extracted to a .wav file for transcription, which is deleted after processing.
        """

        base_chunk = DocumentChunk(
            content=f"Video file: {path.name}",
            source=str(path),
            metadata={"type": "video", "duration": None, "resolution": None},
        )

        if not self.whisper_transcribe:
            return [base_chunk]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_audio = Path(tmp_file.name)

        try:
            self.extract_audio(path, tmp_audio)
            transcript = self.transcribe(tmp_audio)
        finally:
            tmp_audio.unlink(missing_ok=True)

        return [
            base_chunk,
            DocumentChunk(
                content=transcript, source=str(path), metadata={"type": "video"}
            ),
        ]
