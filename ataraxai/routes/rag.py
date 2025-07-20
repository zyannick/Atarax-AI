from pydantic import BaseModel, Field
from typing import List
from fastapi import FastAPI
from ataraxai.routes.status import Status


class CheckManifestResponse(BaseModel):
    status: Status = Field(..., description="Status of the manifest check operation.")
    message: str = Field(
        ..., description="Detailed message about the manifest check operation."
    )


class DirectoriesToScanRequest(BaseModel):
    directories: List[str] = Field(
        ...,
        description="List of directories to scan and index for RAG.",
    )


class DirectoriesToAddRequest(BaseModel):
    directories: List[str] = Field(
        ...,
        description="List of directories to add for RAG indexing.",
    )


class DirectoriesToRemoveRequest(BaseModel):
    directories: List[str] = Field(
        ...,
        description="List of directories to remove from RAG indexing.",
    )


class RebuildIndexResponse(BaseModel):
    status: Status = Field(..., description="Status of the index rebuild operation.")
    message: str = Field(
        ..., description="Detailed message about the index rebuild operation."
    )


class ScanAndIndexResponse(BaseModel):
    status: Status = Field(
        ..., description="Status of the scan and indexing operation."
    )
    message: str = Field(
        ..., description="Detailed message about the scan and indexing operation."
    )


class DirectoriesAdditionResponse(BaseModel):
    status: Status = Field(
        ..., description="Status of the directories addition operation."
    )
    message: str = Field(
        ..., description="Detailed message about the directories addition operation."
    )


class DirectoriesRemovalResponse(BaseModel):
    status: Status = Field(
        ..., description="Status of the directories removal operation."
    )
    message: str = Field(
        ..., description="Detailed message about the directories removal operation."
    )
