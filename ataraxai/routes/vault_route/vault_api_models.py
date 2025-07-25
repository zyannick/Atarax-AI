from pydantic import BaseModel, Field, field_validator, SecretStr
from ataraxai.routes.status import Status

class ConfirmationPhaseRequest(BaseModel):
    confirmation_phrase: str = Field(
        ..., description="The confirmation phrase sent to the user for verification."
    )
    
    @field_validator("confirmation_phrase")
    def validate_confirmation_phrase(cls, v: str) -> str:
        if not v:
            raise ValueError("Confirmation phrase cannot be empty.")
        return v


class ConfirmationPhaseResponse(BaseModel):
    status: Status = Field(..., description="Status of the confirmation phase operation.")
    message: str = Field(
        ..., description="Detailed message about the confirmation phase operation."
    )


class VaultPasswordRequest(BaseModel):
    password: SecretStr = Field(..., min_length=8, description="The user's master password.")
    # class Config:
    #     json_encoders: Dict[Any, Any] = {
    #         SecretStr: lambda v: "***" if "password" in str(v) else v
    #     }
        
    @field_validator("password")
    def validate_password(cls, v: SecretStr) -> SecretStr:
        if v.__len__() < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return v


class VaultPasswordResponse(BaseModel):
    status: Status = Field(..., description="Status of the vault operation.")
    message: str = Field(..., description="Detailed message about the operation.")


class LockVaultResponse(BaseModel):
    status: Status = Field(..., description="Status of the vault lock operation.")
    message: str = Field(
        ..., description="Detailed message about the vault lock operation."
    )
