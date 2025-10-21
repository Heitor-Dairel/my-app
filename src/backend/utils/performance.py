from timeit import timeit
import re


def performance(func: str, setup: str = "main", execution_times: int = 1) -> str:
    r"""
    Measure the execution time of a Python function or statement using `timeit`.

    This function runs the provided code snippet a specified number of times
    and returns the elapsed time in seconds as a formatted string. If `setup`
    is "main", it will automatically import the function from `__main__`.

    Args:
        func (str): The code snippet or function call to be measured (as a string).
        setup (str, optional): Setup code to run before timing. Defaults to "main".
        execution_times (int, optional): Number of times to execute the code snippet.
            Must be greater than 0. Defaults to 1.

    Returns:
        str: The total execution time in seconds, formatted with six decimal places.

    Raises:
        Exception: If `execution_times` is less than 1.
    """

    if execution_times < 1:
        msg = "The number of times the execution will run must be greater than 0"
        raise Exception(msg)

    if setup == "main":
        func_match: re.Match[str] | None = re.match(r"[A-z0-9]*(?=\()", func)

        if func_match:
            setup_avanced: str = f"from __main__ import {func_match.group()}"
            time: float = timeit(stmt=func, setup=setup_avanced, number=execution_times)
            return f"{time:.6f}"

    time: float = timeit(stmt=func, setup=setup, number=execution_times)
    return f"{time:.6f}"
