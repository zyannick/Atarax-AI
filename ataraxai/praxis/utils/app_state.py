from __future__ import annotations
from enum import Enum, auto

class AppState(Enum):
    LOCKED = auto()
    UNLOCKED = auto()
    FIRST_LAUNCH = auto()
    ERROR = auto()