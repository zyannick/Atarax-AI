# in your app_logic, perhaps a preferences_manager.py

import json
from pathlib import Path
from platformdirs import user_config_dir

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"


class PreferencesManager:
    def __init__(self):
        self.config_dir = Path(user_config_dir(appname=APP_NAME, appauthor=APP_AUTHOR))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.preferences_file = self.config_dir / "preferences.json"
        self.preferences = self._load()

    def _default_prefs(self):
        return {
            "watched_directories": [],
            "index_on_startup": True,
            "realtime_monitoring": False,
            "dark_mode": False,
            "llm_model_params": {
                "model_path": "",
                "n_ctx": 2048,
                "n_gpu_layers": 0,
                "main_gpu": 0,
                "tensor_split": False,
                "vocab_only": False,
                "use_map": False,
                "use_mlock": False,
            },
            "generation_params": {
                "n_predict": 128,
                "temp": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "repeat_penalty": 1.1,
                "stop_sequences": [],
                "n_batch": 512,
                "n_threads": 0,
            },
            "whisper_model_params": {
                "model_path": "en",
                "use_gpu": True,
            },
            "whisper_transcription_params": {
                "n_threads": 0,
                "language": "en",
                "translate": False,
                "print_special": False,
                "print_progress": True,
                "no_context": True,
                "max_len": 0,
                "single_segment": False,
                "temperature": 0.0,
            },
        }

    @property
    def preferences_file(self):
        return self.config_dir / "preferences.json"

    @property
    def preferences(self):
        return self._load()

    @preferences.setter
    def preferences(self, new_preferences):
        if isinstance(new_preferences, dict):
            self._preferences = new_preferences
            self.save()
        else:
            raise ValueError("Preferences must be a dictionary.")

    @property
    def watched_directories(self):
        return self.preferences.get("watched_directories", [])

    @watched_directories.setter
    def watched_directories(self, directories):
        if isinstance(directories, list):
            self.preferences["watched_directories"] = directories
            self.save()
        else:
            raise ValueError("Watched directories must be a list.")

    @property
    def index_on_startup(self):
        return self.preferences.get("index_on_startup", True)

    @index_on_startup.setter
    def index_on_startup(self, value):
        if isinstance(value, bool):
            self.preferences["index_on_startup"] = value
            self.save()
        else:
            raise ValueError("Index on startup must be a boolean value.")

    @property
    def realtime_monitoring(self):
        return self.preferences.get("realtime_monitoring", False)

    @realtime_monitoring.setter
    def realtime_monitoring(self, value):
        if isinstance(value, bool):
            self.preferences["realtime_monitoring"] = value
            self.save()
        else:
            raise ValueError("Realtime monitoring must be a boolean value.")

    @property
    def dark_mode(self):
        return self.preferences.get("dark_mode", False)

    @dark_mode.setter
    def dark_mode(self, value):
        if isinstance(value, bool):
            self.preferences["dark_mode"] = value
            self.save()
        else:
            raise ValueError("Dark mode must be a boolean value.")

    @property
    def llm_model_path(self):
        return self.preferences.get("llm_model_path", "")

    @llm_model_path.setter
    def llm_model_path(self, path):
        if isinstance(path, str) and Path(path).is_file():
            self.preferences["llm_model_path"] = path
            self.save()
        else:
            raise ValueError("LLM model path must be a valid file path.")

    @property
    def whisper_model_path(self):
        return self.preferences.get("whisper_model_path", "")

    @whisper_model_path.setter
    def whisper_model_path(self, path):
        if isinstance(path, str) and Path(path).is_file():
            self.preferences["whisper_model_path"] = path
            self.save()
        else:
            raise ValueError("Whisper model path must be a valid file path.")

    def _load(self):
        defaults = self._default_prefs()
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, "r", encoding="utf-8") as f:
                    loaded_prefs = json.load(f)
                defaults.update(loaded_prefs)
                return defaults
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse preferences file {self.preferences_file}. Using defaults."
                )
                return defaults
            except Exception as e:
                print(
                    f"Warning: Error loading preferences {self.preferences_file}: {e}. Using defaults."
                )
                return defaults
        return defaults

    def save(self):
        try:
            with open(self.preferences_file, "w", encoding="utf-8") as f:
                json.dump(self.preferences, f, indent=4)
        except Exception as e:
            print(f"Error saving preferences to {self.preferences_file}: {e}")

    def get(self, key, default_value=None):
        return self.preferences.get(key, default_value)

    def set(self, key, value):
        self.preferences[key] = value
        self.save()
