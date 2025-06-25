from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from pathlib import Path
from typing import List
from ataraxai.app_logic.modules.rag.parser.document_base_parser import DocumentChunk, DocumentParser


class MusicParser(DocumentParser):
    def __init__(
        self,
        whisper_transcribe,
        core_ai_service,
        transcription_params,
    ):
        self.transcribe_audio = whisper_transcribe
        self.core_ai_service = core_ai_service
        self.transcription_params = transcription_params

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribes the given audio file to text.

        Args:
            audio_path (Path): The path to the audio file to be transcribed.

        Returns:
            str: The transcribed text from the audio file.

        Raises:
            ValueError: If the transcription fails and no text is returned.
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
        Parses an MP3 file to extract metadata and optionally transcribe audio to lyrics.

        Args:
            path (Path): The path to the MP3 file to be parsed.

        Returns:
            List[DocumentChunk]: A list containing at least one DocumentChunk with extracted metadata.
                If audio transcription is enabled, a second DocumentChunk containing the transcribed lyrics is included.
        """
        audio = MP3(str(path), ID3=EasyID3)
        tags = {k: v[0] for k, v in audio.items()}
        base_chunk = DocumentChunk(
            content="\n".join([f"{k.title()}: {v}" for k, v in tags.items()]),
            source=str(path),
            metadata={"type": "music", **tags},
        )

        if self.transcribe_audio:
            lyrics = self.transcribe(path)
            lyrics_chunk = DocumentChunk(
                content=lyrics, source=str(path), metadata={"type": "lyrics", **tags}
            )
            return [base_chunk, lyrics_chunk]

        return [base_chunk]
