import logging
import threading
import yaml
from pathlib import Path
from ataraxai.praxis.utils.configs.config_schemas.user_preferences_schema import (
    UserPreferences,
)
from typing_extensions import Optional
from typing import Any

PREFERENCES_FILENAME = "user_preferences.yaml"


class UserPreferencesManager:

    def __init__(
        self,
        config_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the PreferencesManager instance.

        Args:
            config_path (Optional[Path]): The directory path where the preferences file will be stored.
                If None, defaults to the user's home directory under ".ataraxai".

        Attributes:
            config_path (Path): The full path to the preferences file.
            preferences (UserPreferences): The loaded or newly created user preferences.

        Side Effects:
            Creates the preferences directory if it does not exist.
            Loads existing preferences or creates new ones if not found.
        """
        if config_path is None:
            config_path = Path.home() / ".ataraxai"
        self.config_path = config_path / PREFERENCES_FILENAME
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

    def _load_or_create(self) -> UserPreferences:
        """
        Loads user preferences from the configuration file if it exists; otherwise, creates and saves default preferences.

        Returns:
            UserPreferences: The loaded or newly created user preferences object.

        Side Effects:
            - Prints error messages if loading fails.
            - Prints info message if default preferences are used.
            - Saves default preferences to the configuration file if it does not exist or loading fails.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return UserPreferences(**data)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load preferences: {e}")
        self._preferences = UserPreferences()
        self._save()
        return self._preferences

    @property
    def preferences(self) -> UserPreferences:
        """
        Returns the current user preferences.

        Returns:
            UserPreferences: The current user preferences object.
        """
        with self._lock:
            if not hasattr(self, "_preferences"):
                self._preferences = self._load_or_create()
            return self._preferences

    def _save(self):
        """
        Saves the current preferences to the configuration file in YAML format.

        Opens the file specified by `self.config_path` in write mode and writes the
        serialized preferences using YAML. The preferences are obtained by calling
        `model_dump()` on the `self.preferences` object.

        Raises:
            OSError: If the file cannot be opened or written to.
            yaml.YAMLError: If serialization to YAML fails.
        """
        data_to_save = self.preferences.model_dump(mode="json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_to_save, f)

    def update_user_preferences(self, new_prefs: UserPreferences):
        """
        Update the current user preferences with new values and save them.

        Args:
            new_prefs (UserPreferences): The new user preferences to be set.

        Side Effects:
            Updates the internal preferences attribute and persists the changes by calling the _save() method.
        """
        with self._lock:
            if not isinstance(new_prefs, UserPreferences):
                raise TypeError("new_prefs must be an instance of UserPreferences")
            self._preferences = new_prefs
            self._save()

    def get(self, key: str, default=None) -> Any:  # type: ignore
        """
        Retrieve the value of a preference by key.

        Args:
            key (str): The name of the preference to retrieve.
            default (Optional[Any]): The value to return if the preference is not found. Defaults to None.

        Returns:
            Union[int, str, bool]: The value of the preference if found, otherwise the default value.
        """
        return getattr(self.preferences, key, default)  # type: ignore

    def set(self, key: str, value: Any) -> None:
        """
        Set a preference value for the given key and persist the change.

        Args:
            key (str): The name of the preference to set.
            value (Union[str, int, bool, Dict[str, Any]]): The value to assign to the preference.

        Raises:
            AttributeError: If the key does not correspond to a valid preference attribute.

        Side Effects:
            Updates the preferences object and saves the changes to persistent storage.
        """
        if not hasattr(self.preferences, key):
            raise AttributeError(f"Preference '{key}' does not exist.")
        setattr(self.preferences, key, value)
        self._save()

    def reload(self):
        """
        Reloads the user preferences by reloading them from the storage or creating them if they do not exist.

        This method updates the `preferences` attribute with the latest preferences data.
        """
        with self._lock:
            self._preferences = self._load_or_create()
