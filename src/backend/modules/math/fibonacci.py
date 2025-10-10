from src.backend.helpers.scientific_notation import format_number


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
