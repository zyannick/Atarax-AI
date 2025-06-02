import subprocess
from pathlib import Path
from typing import List
from .document_base_parser import DocumentChunk, DocumentParser
import tempfile


class VideoParser(DocumentParser):
    def __init__(self, whisper_path="./main", model_path="models/ggml-base.en.bin"):
        self.whisper_path = whisper_path
        self.model_path = model_path

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
        cmd = [self.whisper_path, "-m", self.model_path, "-f", str(audio_path), "-otxt"]
        subprocess.run(cmd, check=True)
        txt_path = audio_path.with_suffix(".txt")
        return txt_path.read_text(encoding="utf-8")

    def parse(self, path: Path) -> List[DocumentChunk]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_audio = Path(tmp_file.name)

        try:
            self.extract_audio(path, tmp_audio)
            transcript = self.transcribe(tmp_audio)
        finally:
            tmp_audio.unlink(missing_ok=True)

        return [
            DocumentChunk(
                content=transcript, source=str(path), metadata={"type": "video"}
            )
        ]
