from pydantic import BaseModel
from typing import List, Dict, Any
from ataraxai.routes.status import Status


class StartChainResponse(BaseModel):
    status: str
    message: str
    task_id: str


class RunChainRequest(BaseModel):
    """
    Request model for running a chain of tasks.

    Attributes:
        chain_definition (List[Dict[str, Any]]): A list of task definitions that make up the chain.
        initial_user_query (str): The initial user query to start the chain execution.
    """

    chain_definition: List[Dict[str, Any]]
    initial_user_query: str

    class Config:
        """
        Pydantic configuration to allow arbitrary types in the chain definition.
        """

        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return f"RunChainRequest(chain_definition={self.chain_definition}, initial_user_query={self.initial_user_query})"


class RunChainResponse(BaseModel):
    status: Status
    message: str
    result: Any


class CancelChainResponse(BaseModel):
    status: Status
    message: str


class AvailableTasksResponse(BaseModel):

    status: Status
    message: str
    list_available_tasks: List[Dict[str, Any]]

    def __str__(self) -> str:
        return (
            f"AvailableTasksResponse(list_available_tasks={self.list_available_tasks})"
        )
