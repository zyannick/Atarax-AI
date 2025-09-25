from typing import Any, Dict
from pydantic import BaseModel, Field

from ataraxai.gateway.gateway_task_manager import TaskStatus


# class Status(str, Enum):
#     SUCCESS = auto()
#     FAILURE = auto()
#     PENDING = auto()
#     ERROR = auto()

Status = TaskStatus


class StatusResponse(BaseModel):
    status: TaskStatus = Field(..., description="Status of the operation.")
    message: str = Field(..., description="Detailed message about the operation.")
    task_id: str = Field(default_factory=str, description="ID of the task associated with the operation, if applicable.")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional data related to the status response.")
