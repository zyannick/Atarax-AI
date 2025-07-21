from pydantic import BaseModel, Field, field_validator, validator
from typing import List
from fastapi import FastAPI
from ataraxai.routes.status import Status
from pathlib import Path


class CheckManifestResponse(BaseModel):
    status: Status = Field(..., description="Status of the manifest check operation.")
    message: str = Field(
        ..., description="Detailed message about the manifest check operation."
    )

    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        if not v:
            raise ValueError("Message cannot be empty.")
        return v


class DirectoriesToScanRequest(BaseModel):
    directories: List[str] = Field(
        ...,
        description="List of directories to scan and index for RAG.",
    )
    
    @field_validator("directories")
    def validate_directories(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Directories list cannot be empty.")
        for directory in v:
            if not directory.strip():
                raise ValueError("Directory paths cannot be empty or whitespace.")
        return v


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
