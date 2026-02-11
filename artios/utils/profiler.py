from functools import wraps

from torch.profiler import record_function


def profile_function(label=None):
    """Decorator that only profiles if a profiler is active

    Args:
        label: label to use for the function

    Example usage:
    ```python
    #@profile_function(label="example_train")
    def example_train():
        pass
    ```
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            fn_label = label or f"{func.__module__}.{func.__name__}"
            with record_function(fn_label):
                return func(*args, **kwargs)

        return wrapper

    return decorator
