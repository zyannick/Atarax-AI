import functools
from fastapi import  HTTPException,  status
from ataraxai.praxis.utils.exceptions import AtaraxAIError




def handle_api_errors(operation_name: str, logger=None):
    def decorator(func):
        @functools.wraps(func) # Preserve the original function's metadata
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except ValueError as e:
                if logger:
                    logger.warning(f"Validation error in {operation_name}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )
            except KeyError as e:
                if logger:
                    logger.error(f"Missing key in {operation_name}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Required item not found: {e}",
                )
            except AtaraxAIError as e:
                if logger:
                    logger.error(f"{operation_name} failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"{operation_name} failed: {str(e)}",
                )
            except Exception as e:
                if logger:
                    logger.error(f"Unexpected error in {operation_name}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"{operation_name} failed due to an unexpected error",
                )

        return wrapper

    return decorator
