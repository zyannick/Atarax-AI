from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from pathlib import Path
from typing import List
from .document_base_parser import DocumentChunk, DocumentParser
import subprocess


class MusicParser(DocumentParser):
    def __init__(
        self,
        whisper_transcribe=False,
        whisper_path="./main",
        model_path="models/ggml-base.en.bin",
    ):
        # TODO : import the c++ module for whisper rather than using subprocess
        self.transcribe_audio = whisper_transcribe
        self.whisper_path = whisper_path
        self.model_path = model_path

    def transcribe(self, path: Path) -> str:
        cmd = [self.whisper_path, "-m", self.model_path, "-f", str(path), "-otxt"]
        subprocess.run(cmd, check=True)
        return path.with_suffix(".txt").read_text(encoding="utf-8")

    def parse(self, path: Path) -> List[DocumentChunk]:
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
