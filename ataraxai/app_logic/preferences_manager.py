import yaml
from pathlib import Path
from ataraxai.app_logic.utils.config_schemas.user_preferences_schema import (
    UserPreferences,
)
from typing_extensions import Optional
from typing import Dict, Any, Union

PREFERENCES_FILENAME = "user_preferences.yaml"


class PreferencesManager:

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path.home() / ".ataraxai"
        self.config_path = config_path / PREFERENCES_FILENAME
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.preferences: UserPreferences = self._load_or_create()

    def _load_or_create(self) -> UserPreferences:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return UserPreferences(**data)
            except Exception as e:
                print(f"[ERROR] Failed to load preferences: {e}")
        print("[INFO] Using default user preferences.")
        prefs = UserPreferences()
        self._save(prefs)
        return prefs

    def _save(self, prefs: Optional[UserPreferences] = None):
        prefs = prefs or self.preferences
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(prefs.model_dump(), f)

    def get(self, key: str, default=None):  # type: ignore
        return getattr(self.preferences, key, default)  # type: ignore

    def set(self, key: str, value: Union[str, int, bool, Dict[str, Any]]):
        setattr(self.preferences, key, value)
        self._save()

    def reload(self):
        self.preferences = self._load_or_create()
