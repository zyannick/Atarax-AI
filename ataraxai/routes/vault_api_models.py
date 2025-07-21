from pydantic import BaseModel, Field
from typing import Dict, Any    
from ataraxai.hegemonikon_py import SecureString as _SecureString  # type: ignore
from ataraxai.routes.status import Status

class ConfirmationPhaseRequest(BaseModel):
    confirmation_phrase: str = Field(
        ..., description="The confirmation phrase sent to the user for verification."
    )


class ConfirmationPhaseResponse(BaseModel):
    status: Status = Field(..., description="Status of the confirmation phase operation.")
    message: str = Field(
        ..., description="Detailed message about the confirmation phase operation."
    )


class VaultPasswordRequest(BaseModel):
    password: str = Field(..., min_length=8, description="The user's master password.")
    class Config:
        json_encoders: Dict[Any, Any] = {
            str: lambda v: "***" if "password" in str(v) else v
        }


class VaultPasswordResponse(BaseModel):
    status: Status = Field(..., description="Status of the vault operation.")
    message: str = Field(..., description="Detailed message about the operation.")


class LockVaultResponse(BaseModel):
    status: Status = Field(..., description="Status of the vault lock operation.")
    message: str = Field(
        ..., description="Detailed message about the vault lock operation."
    )
