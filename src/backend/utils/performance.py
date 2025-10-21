from timeit import timeit
import re


def performance(func: str, setup: str = "main", execution_times: int = 1) -> str:

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
