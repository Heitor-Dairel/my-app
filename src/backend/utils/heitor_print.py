from rich.console import Console
from rich import get_console
from typing import IO, Any, Callable
from src.backend.constants import COLORS, STYLE_TEXT


class HDPrint:
    r"""
    Enhanced printing utility for styled console output and JSON formatting.

    This class allows printing Python objects to the console with optional
    text styles, foreground and background colors, and also provides a
    convenient method to print JSON-formatted representations.
    """

    def __init__(self, *objects: Any) -> None:
        """
        Initialize the HDPrint instance with objects to be printed.

        Args:
            *objects (Any): One or more Python objects to print.
        """

        self.objects = objects

    def print(
        self,
        *,
        sep: str = " ",
        end: str = "\n",
        file: IO[str] | None = None,
        style_text: STYLE_TEXT | None = None,
        foreground: COLORS | None = None,
        background: COLORS | None = None,
    ) -> None:
        r"""
        Print objects to the console with custom styling.

        Args:
            sep (str, optional): Separator between objects. Defaults to " ".
            end (str, optional): String appended after the last object. Defaults to "\n".
            file (IO[str] | None, optional): File or stream to write output to.
                Defaults to None (prints to console).
            style_text (STYLE_TEXT | None, optional): Text style (bold, italic, etc.).
            foreground (COLORS | None, optional): Foreground color of the text.
            background (COLORS | None, optional): Background color of the text.

        Process:
            1. Build a style string combining style, foreground, and background.
            2. Determine the output target (console or file).
            3. Print the objects using the `rich` Console with the style applied.

        Returns:
            None
        """

        parts: list[str] = []

        if style_text is not None:
            parts.append(str(style_text))

        if foreground is not None:
            parts.append(str(foreground))

        if background is not None:
            parts.append("on " + str(background))

        style: str | None = " ".join(parts)

        write_console: Console = get_console() if file is None else Console(file=file)
        return write_console.print(*self.objects, style=style, sep=sep, end=end)

    def print_json(
        self,
        *,
        indent: None | int | str = 2,
        highlight: bool = True,
        skip_keys: bool = False,
        ensure_ascii: bool = False,
        check_circular: bool = True,
        allow_nan: bool = True,
        default: Callable[[Any], Any] | None = None,
        sort_keys: bool = False,
    ) -> None:
        r"""
        Print objects as JSON-formatted strings using rich's console.

        Args:
            data (Any, optional): JSON data to print. Defaults to None (uses self.objects).
            indent (int | str | None, optional): Indentation for JSON output. Defaults to 2.
            highlight (bool, optional): Enable syntax highlighting. Defaults to True.
            skip_keys (bool, optional): Skip keys that are not serializable. Defaults to False.
            ensure_ascii (bool, optional): Escape non-ASCII characters. Defaults to False.
            check_circular (bool, optional): Check for circular references. Defaults to True.
            allow_nan (bool, optional): Allow NaN, Infinity, -Infinity. Defaults to True.
            default (Callable[[Any], Any] | None, optional): Function to serialize unsupported objects.
            sort_keys (bool, optional): Sort dictionary keys. Defaults to False.

        Process:
            1. Serialize `self.objects` or `data` into JSON string.
            2. Print JSON to the console with optional highlighting and formatting.

        Returns:
            None
        """

        get_console().print_json(
            data=self.objects,
            indent=indent,
            highlight=highlight,
            skip_keys=skip_keys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            default=default,
            sort_keys=sort_keys,
        )


if __name__ == "__main__":

    data = {"nome": "Heitor", "idade": 22, "linguagens": ["Python", "SQL"]}

    HDPrint(data).print_json()

    HDPrint("ola", 1, 2, 3, 4).print(sep="\n", foreground="black")

    # python -W ignore -m src.backend.utils.heitor_print
