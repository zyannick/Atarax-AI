from __future__ import annotations
from enum import Enum, auto

class AppState(Enum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    FIRST_LAUNCH = "first_launch"
    ERROR = "error"