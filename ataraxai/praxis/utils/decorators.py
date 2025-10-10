import functools
import inspect
import logging
import traceback
from typing import Optional

from fastapi import HTTPException, status

from ataraxai.praxis.utils.exceptions import AtaraxAIError


def handle_api_errors(operation_name: str):
    def decorator(func):
        @functools.wraps(func)  # Preserve the original function's metadata
        async def wrapper(*args, **kwargs):
            logger: Optional[logging.Logger] = kwargs.get('logger')
            
            if logger is None:
                try:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    
                    if 'logger' in param_names:
                        logger_index = param_names.index('logger')
                        if logger_index < len(args):
                            potential_logger = args[logger_index]
                            if isinstance(potential_logger, logging.Logger):
                                logger = potential_logger
                except Exception:
                    pass
                
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except ValueError as e:
                if logger:
                    logger.warning(f"Validation error in {operation_name}: {e}")
                    logger.debug(traceback.format_exc())
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )
            except KeyError as e:
                if logger:
                    logger.error(f"Missing key in {operation_name}: {e}")
                    logger.debug(traceback.format_exc())
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Required item not found: {e}",
                )
            except AtaraxAIError as e:
                if logger:
                    logger.error(f"{operation_name} failed: {e}")
                    logger.debug(traceback.format_exc())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"{operation_name} failed: {str(e)}",
                )
            except Exception as e:
                if logger:
                    logger.error(f"Unexpected error in {operation_name}: {e}")
                    logger.debug(traceback.format_exc())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"{operation_name} failed due to an unexpected error",
                )

        return wrapper

    return decorator
