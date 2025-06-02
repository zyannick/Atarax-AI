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
            "llm_model_path": "", 
            "whisper_model_path": "",
        }

    def _load(self):
        defaults = self._default_prefs()
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r', encoding='utf-8') as f:
                    loaded_prefs = json.load(f)
                defaults.update(loaded_prefs)
                return defaults
            except json.JSONDecodeError:
                print(f"Warning: Could not parse preferences file {self.preferences_file}. Using defaults.")
                return defaults
            except Exception as e:
                print(f"Warning: Error loading preferences {self.preferences_file}: {e}. Using defaults.")
                return defaults
        return defaults

    def save(self):
        try:
            with open(self.preferences_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=4)
        except Exception as e:
            print(f"Error saving preferences to {self.preferences_file}: {e}")


    def get(self, key, default_value=None):
        return self.preferences.get(key, default_value)

    def set(self, key, value):
        self.preferences[key] = value
        self.save()

# Example Usage:
# if __name__ == "__main__":
#     prefs_manager = PreferencesManager()
#     print(f"Config file location: {prefs_manager.preferences_file}")
#     print(f"Current watched directories: {prefs_manager.get('watched_directories')}")
#
#     # Example: Add a directory to watch
#     # current_dirs = prefs_manager.get('watched_directories', [])
#     # if "/home/yzoetgna/Documents" not in current_dirs:
#     #     current_dirs.append("/home/yzoetgna/Documents")
#     #     prefs_manager.set('watched_directories', current_dirs)
#     #     print(f"Updated watched directories: {prefs_manager.get('watched_directories')}")
#
#     prefs_manager.set("dark_mode", True)
#     print(f"Dark mode: {prefs_manager.get('dark_mode')}")