from enum import Enum, auto


class ServiceStatus(Enum):
    NOT_INITIALIZED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    FAILED = auto()
