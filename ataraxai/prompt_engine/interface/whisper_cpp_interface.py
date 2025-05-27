import subprocess
import os
import tempfile 

class WhisperInterface:
    def __init__(self, executable_path: str, model_path: str,
                 ffmpeg_path: str = "ffmpeg", 
                 default_params: dict = None):
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"Whisper.cpp executable not found at: {executable_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Whisper model file not found at: {model_path}")

        self.executable_path = executable_path
        self.model_path = model_path
        self.ffmpeg_path = ffmpeg_path

        self._check_ffmpeg() 

        self._base_default_params = {
            "language": "auto",
            "threads": os.cpu_count() or 4,
            "output_format": "txt",
            "print_special": False,
            "no_timestamps": True,
            "timeout_ffmpeg": 60, # Timeout for ffmpeg conversion in seconds
            "timeout_whisper": 300, # Timeout for whisper transcription in seconds
        }
        if default_params:
            self._base_default_params.update(default_params)
        self.default_params = self._base_default_params

    def _check_ffmpeg(self):
        try:
            process = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True, text=True, check=True, timeout=5
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ffmpeg executable not found at '{self.ffmpeg_path}'. "
                "Please install ffmpeg and ensure it's in your PATH or provide the correct path."
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                f"ffmpeg at '{self.ffmpeg_path}' does not seem to be working correctly. "
                f"Error: {e.stderr if hasattr(e, 'stderr') else e}"
            )

    def _preprocess_audio_with_ffmpeg(self, input_audio_path: str, params: dict) -> str:
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Input audio file for ffmpeg not found: {input_audio_path}")

        temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav_file.name
        temp_wav_file.close()

        ffmpeg_command = [
            self.ffmpeg_path,
            "-i", input_audio_path,
            "-ar", "16000",
            "-ac", "1",
            "-loglevel", "error",
            "-y",
            temp_wav_path
        ]

        try:
            print(f"Running ffmpeg: {' '.join(ffmpeg_command)}")
            subprocess.run(
                ffmpeg_command,
                check=True, capture_output=True, text=True,
                timeout=params.get("timeout_ffmpeg", 60)
            )
            return temp_wav_path
        except subprocess.CalledProcessError as e:
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
            error_message = (f"ffmpeg processing failed for {input_audio_path} with code {e.returncode}.\n"
                             f"Command: {' '.join(e.cmd)}\nStderr: {e.stderr}")
            print(error_message)
            raise RuntimeError(error_message) from e
        except subprocess.TimeoutExpired as e:
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
            error_message = f"ffmpeg processing timed out for {input_audio_path}.\nCommand: {' '.join(ffmpeg_command)}"
            print(error_message)
            raise TimeoutError(error_message) from e
        except Exception as e:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            raise

    def _build_whisper_cli_command(self, audio_to_transcribe_path: str, params: dict) -> list:
        cmd = [
            self.executable_path,
            "-m", self.model_path,
            "-f", audio_to_transcribe_path,
            "-l", params.get("language", "auto"),
            "-t", str(params.get("threads", 4)),
        ]
        output_format = params.get("output_format", "txt").lower()
        if output_format == "txt":
            cmd.append("-otxt")
            if params.get("no_timestamps", True): cmd.append("-nt")
        elif output_format == "json": cmd.append("-oj")
        elif output_format == "srt": cmd.append("-osrt")
        elif output_format == "vtt": cmd.append("-ovtt")

        if params.get("print_special", False): cmd.append("-ps")
        if params.get("translate", False): cmd.append("-tr")
        if params.get("word_timestamps", False): cmd.append("-owts")
        return cmd

    def transcribe(self, audio_file_path: str, specific_params: dict = None) -> str:
        current_params = self.default_params.copy()
        if specific_params:
            current_params.update(specific_params)

        temp_ffmpeg_output_file = None
        try:
            print(f"Preprocessing audio with ffmpeg: {audio_file_path}")
            temp_ffmpeg_output_file = self._preprocess_audio_with_ffmpeg(audio_file_path, current_params)
            audio_to_transcribe = temp_ffmpeg_output_file
            print(f"ffmpeg processed audio to: {audio_to_transcribe}")
            
            command = self._build_whisper_cli_command(audio_to_transcribe, current_params)
            
            output_filename_base = os.path.splitext(audio_to_transcribe)[0]
            output_format_ext = current_params.get("output_format", "txt").lower()
            expected_whisper_output_file = f"{output_filename_base}.{output_format_ext}"

            if os.path.exists(expected_whisper_output_file):
                try: os.remove(expected_whisper_output_file)
                except OSError as e: print(f"Warning: Could not remove pre-existing whisper output file {expected_whisper_output_file}: {e}")

            print(f"Running whisper.cpp: {' '.join(command)}")
            process = subprocess.run(
                command,
                capture_output=True, text=True, check=True,
                timeout=current_params.get("timeout_whisper", 300)
            )
            
            if os.path.exists(expected_whisper_output_file):
                with open(expected_whisper_output_file, 'r', encoding='utf-8') as f:
                    transcribed_text = f.read().strip()
                return transcribed_text
            else:
                raise RuntimeError(f"Whisper.cpp did not create the expected output file: {expected_whisper_output_file}\n"
                                   f"Stdout: {process.stdout}\nStderr: {process.stderr}")

        except subprocess.CalledProcessError as e:
            error_message = (f"Whisper.cpp (CLI) execution failed with code {e.returncode}.\n"
                             f"Command: {' '.join(e.cmd)}\nStdout: {e.stdout}\nStderr: {e.stderr}")
            print(error_message)
            raise RuntimeError(error_message) from e
        except subprocess.TimeoutExpired as e:
            error_message = f"Whisper.cpp (CLI) execution timed out.\nCommand: {' '.join(command if 'command' in locals() else 'N/A')}" # command might not be defined if ffmpeg fails first
            print(error_message)
            raise TimeoutError(error_message) from e
        except Exception as e: 
            print(f"An error occurred during transcription: {e}")
            raise 
        finally:
            if 'expected_whisper_output_file' in locals() and os.path.exists(expected_whisper_output_file):
                try: os.remove(expected_whisper_output_file)
                except OSError as e: print(f"Warning: Could not remove whisper output file {expected_whisper_output_file}: {e}")
            
            if temp_ffmpeg_output_file and os.path.exists(temp_ffmpeg_output_file):
                try:
                    print(f"Cleaning up temporary ffmpeg file: {temp_ffmpeg_output_file}")
                    os.remove(temp_ffmpeg_output_file)
                except OSError as e: print(f"Warning: Could not remove temporary ffmpeg file {temp_ffmpeg_output_file}: {e}")
        

        return "" 
