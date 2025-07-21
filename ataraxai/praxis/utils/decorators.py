from fastapi import  HTTPException,  status
from ataraxai.praxis.utils.exceptions import AtaraxAIError




def handle_api_errors(operation_name: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except AtaraxAIError as e:
                # logger.error(f"{operation_name} failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"{operation_name} failed: {str(e)}",
                )
            except Exception as e:
                # logger.error(f"Unexpected error in {operation_name}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"{operation_name} failed due to an unexpected error",
                )

        return wrapper

    return decorator
