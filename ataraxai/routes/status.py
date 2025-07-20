from pydantic import BaseModel, Field
from enum import Enum, auto


class Status(str, Enum):
    SUCCESS = auto()
    FAILURE = auto()
    PENDING = auto()
    ERROR = auto()


class StatusResponse(BaseModel):
    status: Status = Field(..., description="Status of the operation.")
    message: str = Field(..., description="Detailed message about the operation.")
