"""Error handling utilities for RolyPoly.

Decorators and utilities for handling errors and exceptions
in a consistent way across the RolyPoly codebase (but really mostly in filter-reads? ) #TODO: check if obsolete.

Example:
    ```python
    @handle_errors
    def my_function():
        # Function code that might raise exceptions
        pass
    ```
"""

import functools
import logging
from typing import Callable, Any

def handle_errors(func: Callable) -> Callable:
    """Decorator that handles exceptions and logs them appropriately.

    This decorator wraps a function and catches any exceptions that occur during
    its execution. It logs the exception details using the provided logger or
    creates a new one if none is provided.

    Args:
        func (Callable): The function to wrap with error handling

    Returns:
        Callable: The wrapped function with error handling

    Example:
        ```python
        @handle_errors
        def risky_function(logger, x, y):
            return x / y
        
        risky_function(logging.getLogger(), 1, 0)
        # Will log the ZeroDivisionError and re-raise it
        ```
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Assuming the first argument is always the logger
            logger = args[0] if args and isinstance(args[0], logging.Logger) else logging.getLogger(__name__)
            logger.exception(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

# def retry(max_attempts: int, logger: logging.Logger) -> Callable:
#     """
#     A decorator that retries the wrapped function a specified number of times before giving up.

#     Args:
#         max_attempts (int): The maximum number of attempts to make.
#         logger (logging.Logger): The logger object to use for logging retries and errors.

#     Returns:
#         Callable: The decorated function.
#     """
#     def decorator(func: Callable) -> Callable:
#         @functools.wraps(func)
#         def wrapper(*args: Any, **kwargs: Any) -> Any:
#             attempts = 0
#             while attempts < max_attempts:
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     attempts += 1
#                     logger.warning(f"Attempt {attempts} of {max_attempts} failed for {func.__name__}: {str(e)}")
#                     if attempts == max_attempts:
#                         logger.error(f"All {max_attempts} attempts failed for {func.__name__}. Final exception: {str(e)}")
#                         logger.error(f"Traceback: {traceback.format_exc()}")
#                         raise
#         return wrapper
#     return decorator

# # def validate_input(validator: Callable) -> Callable:
# #     """
# #     A decorator that validates the input of a function using a provided validator function.

# #     Args:
#         validator (Callable): A function that takes the same arguments as the decorated function
#                               and returns True if the input is valid, False otherwise.

#     Returns:
#         Callable: The decorated function.
#     """
#     def decorator(func: Callable) -> Callable:
#         @functools.wraps(func)
#         def wrapper(*args: Any, **kwargs: Any) -> Any:
#             if not validator(*args, **kwargs):
#                 raise ValueError(f"Invalid input for {func.__name__}")
#             return func(*args, **kwargs)
#         return wrapper
#     return decorator

# def log_execution_time(logger: logging.Logger) -> Callable:
#     """
#     A decorator that logs the execution time of a function.

#     Args:
#         logger (logging.Logger): The logger object to use for logging the execution time.

#     Returns:
#         Callable: The decorated function.
#     """
#     def decorator(func: Callable) -> Callable:
#         @functools.wraps(func)
#         def wrapper(*args: Any, **kwargs: Any) -> Any:
#             import time
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             end_time = time.time()
#             logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
#             return result
#         return wrapper
#     return decorator

# # # Example usage
# # if __name__ == "__main__":
# #     from logging_setup import setup_logging
# #     logger = setup_logging("error_handling_test.log")

# #     @handle_errors(logger)
# #     def example_function(x: int, y: int) -> int:
# #         return x / y

# #     @retry(max_attempts=3, logger=logger)
# #     def unreliable_function() -> None:
# #         import random
# #         if random.random() < 0.8:
# #             raise ValueError("Random error occurred")
# #         print("Function succeeded")

# #     def validate_positive(x: int, y: int) -> bool:
# #         return x > 0 and y > 0

# #     @validate_input(validate_positive)
# #     def divide_positive_numbers(x: int, y: int) -> float:
# #         return x / y

# #     @log_execution_time(logger)
# #     def time_consuming_function() -> None:
# #         import time
# #         time.sleep(2)

#     # Test handle_errors
#     try:
#         example_function(5, 0)
#     except ZeroDivisionError:
#         print("Caught ZeroDivisionError as expected")

#     # Test retry
#     unreliable_function()

#     # Test validate_input
#     try:
#         divide_positive_numbers(5, -1)
#     except ValueError:
#         print("Caught ValueError as expected")

#     # Test log_execution_time
#     time_consuming_function()

#     print("All tests completed. Check the log file for details.")