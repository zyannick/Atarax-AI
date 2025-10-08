from enum import Enum, auto


class ServiceStatus(Enum):
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"
