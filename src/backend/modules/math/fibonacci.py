from decimal import Decimal
from src.backend.utils import performance, HDPrint


def format_number(num: int, places: int = 2, start_range: int = 100) -> str:
    r"""
    Format a number as a string, using a thousands separator or scientific notation.

    This function formats small numbers with a dot as a thousands separator,
    and switches to scientific notation for very large numbers or zero.

    Args:
        num (int): The number to format.
        places (int, optional): Number of decimal places for scientific notation. Defaults to 2.
        start_range (int, optional): Minimum number of digits to trigger scientific notation. Defaults to 100.

    Returns:
        str: Formatted number string.

    Examples:
        >>> format_number(123456)
        '123.456'
        >>> format_number(0)
        '0.00e+0'
        >>> format_number(10**120)
        '1.00e+120'
    """

    num_1: Decimal = Decimal(num)

    if num_1 == 0:
        return f"{0:.{places}e}"

    exp: int = num_1.adjusted()

    if exp + 1 > start_range:

        mantissa: Decimal = num_1.scaleb(-exp)
        return f"{mantissa:.{places}f}e{exp:+d}"

    return f"{num:,}".replace(",", ".")


def fibonacci_serie(number: int = 10, single: bool = False) -> list[str] | str | None:
    r"""
    Generate Fibonacci sequence values formatted as strings.

    This function computes Fibonacci numbers up to a given position and formats
    each value using the `format_number` function. It supports returning either
    the full sequence or a single Fibonacci number, depending on the `single` parameter.

    Args:
        number (int, optional): Number of terms to generate in the sequence.
            Must be between 1 and 20,000. Defaults to 10.
        single (bool, optional):
            - If `False`, returns a list with the complete sequence up to `number`.
            - If `True`, returns only the last Fibonacci number calculated.
            Defaults to `False`.

    Raises:
        Exception: If `number` is negative or greater than 20,000.

    Returns:
        (list[str] | str | None):
            - `list[str]`: Full sequence of formatted Fibonacci numbers if `single=False`.
            - `str`: Single formatted Fibonacci number if `single=True`.
            - `None`: If `number` equals 0.

    Example:
        ```
        fibonacci_serie(5)
        ['0', '1', '1', '2', '3']

        fibonacci_serie(5, single=True)
        '3'
        ```
    """

    if number > 20000:
        msg: str = f"Number {number!r} out of range."
        raise Exception(msg)

    if number < 0:
        msg: str = f"The number entered is negative, invalid number {number!r}."
        raise Exception(msg)

    if number == 0:
        return None

    a: int = 0
    b: int = 1
    result: str
    lista: list[str] = []

    if not single:
        for _ in range(number):
            lista.append(format_number(a, places=2))
            a, b = a + b, a
        return lista

    for _ in range(number):
        result = format_number(a, places=2)
        a, b = a + b, a

    return result


if __name__ == "__main__":

    HDPrint(
        performance(
            func="fibonacci_serie(1000, single=True)",
            execution_times=1,
        ),
        fibonacci_serie(1000, single=True),
    ).print()

# 5.405773 funcao antiga

# python -W ignore -m src.backend.modules.math.fibonacci
