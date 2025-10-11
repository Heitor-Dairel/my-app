from decimal import Decimal


def format_number(num: int, places: int = 2, start_range: int = 100) -> str:
    r"""
    Format a number either in scientific notation or as a locale-style string based on its length.

    Process:
        1. Convert the input number to a string and check its length.
        2. If the number's string length exceeds `start_range`:
            - Convert it to a `Decimal` for precise handling.
            - If the value is zero, return `"0.00e+00"` (depending on `places`).
            - Compute the exponent and mantissa manually.
            - Format the mantissa to the given decimal places and combine it with the exponent.
        3. If within range, return the number formatted with dots as thousands separators (e.g., `"1.000.000"`).

    Args:
        num (int): The number to format.
        places (int, optional): Number of decimal places for scientific notation. Defaults to 2.
        start_range (int, optional): Threshold for switching to scientific notation. Defaults to 100.

    Returns:
        str:
            - Scientific notation (e.g., `"1.23e+05"`) for large numbers.
            - Dotted thousands format (e.g., `"1.000.000"`) for smaller numbers.

    Example:
        ```
        format_number(12345678901234567890)
        '1.23e+19'
        format_number(4234832753845293, places=4, start_range=0)
        '4.2348e+15'
        format_number(123456)
        '123.456'
        format_number(0)
        '0.00e+00'
        ```

    Notes:
        - Uses the `Decimal` class to maintain precision for large numbers.
        - The thousands separator uses `'.'` instead of a comma for visual consistency.
    """

    if len(str(num)) > start_range:
        num_1: Decimal = Decimal(num)
        if num_1 == 0:
            return f"{0:.{places}e}"
        exp: int = num_1.adjusted()
        mantissa: Decimal = num_1.scaleb(-exp).normalize()
        mantissa_str: str = f"{mantissa:.{places}f}"
        return f"{mantissa_str}e{exp:+d}"

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
        return (list[str] | str | None):
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

    a: int = 0
    b: int = 1
    result: str | None = None
    lista: list[str] = []

    if number > 20000:
        msg: str = f"Number {number!r} out of range."
        raise Exception(msg)

    if number < 0:
        msg: str = f"The number entered is negative, invalid number {number!r}."
        raise Exception(msg)

    if number == 0:
        return result

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

    print(type(fibonacci_serie(190, single=True)))
    print(fibonacci_serie(190, single=True))

    # python -W ignore -m src.backend.modules.math.fibonacci
