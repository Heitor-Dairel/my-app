from rich.console import Console
from rich import get_console
from typing import IO, Any, Callable, Literal


class HDPrint:
    r"""
    Enhanced printing utility for styled console output and JSON formatting.

    This class allows printing Python objects to the console with optional
    text styles, foreground and background colors, and also provides a
    convenient method to print JSON-formatted representations.
    """

    COLORS = Literal[
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "bright_black",
        "bright_red",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_magenta",
        "bright_cyan",
        "bright_white",
        "grey0",
        "navy_blue",
        "dark_blue",
        "blue3",
        "blue1",
        "dark_green",
        "deep_sky_blue4",
        "dodger_blue3",
        "dodger_blue2",
        "green4",
        "spring_green4",
        "turquoise4",
        "deep_sky_blue3",
        "dodger_blue1",
        "dark_cyan",
        "light_sea_green",
        "deep_sky_blue2",
        "deep_sky_blue1",
        "green3",
        "spring_green3",
        "cyan3",
        "dark_turquoise",
        "turquoise2",
        "green1",
        "spring_green2",
        "spring_green1",
        "medium_spring_green",
        "cyan2",
        "cyan1",
        "purple4",
        "purple3",
        "blue_violet",
        "grey37",
        "medium_purple4",
        "slate_blue3",
        "royal_blue1",
        "chartreuse4",
        "pale_turquoise4",
        "steel_blue",
        "steel_blue3",
        "cornflower_blue",
        "dark_sea_green4",
        "cadet_blue",
        "sky_blue3",
        "chartreuse3",
        "sea_green3",
        "aquamarine3",
        "medium_turquoise",
        "steel_blue1",
        "sea_green2",
        "sea_green1",
        "dark_slate_gray2",
        "dark_red",
        "dark_magenta",
        "orange4",
        "light_pink4",
        "plum4",
        "medium_purple3",
        "slate_blue1",
        "wheat4",
        "grey53",
        "light_slate_grey",
        "medium_purple",
        "light_slate_blue",
        "yellow4",
        "dark_sea_green",
        "light_sky_blue3",
        "sky_blue2",
        "chartreuse2",
        "pale_green3",
        "dark_slate_gray3",
        "sky_blue1",
        "chartreuse1",
        "light_green",
        "aquamarine1",
        "dark_slate_gray1",
        "deep_pink4",
        "medium_violet_red",
        "dark_violet",
        "purple",
        "medium_orchid3",
        "medium_orchid",
        "dark_goldenrod",
        "rosy_brown",
        "grey63",
        "medium_purple2",
        "medium_purple1",
        "dark_khaki",
        "navajo_white3",
        "grey69",
        "light_steel_blue3",
        "light_steel_blue",
        "dark_olive_green3",
        "dark_sea_green3",
        "light_cyan3",
        "light_sky_blue1",
        "green_yellow",
        "dark_olive_green2",
        "pale_green1",
        "dark_sea_green2",
        "pale_turquoise1",
        "red3",
        "deep_pink3",
        "magenta3",
        "dark_orange3",
        "indian_red",
        "hot_pink3",
        "hot_pink2",
        "orchid",
        "orange3",
        "light_salmon3",
        "light_pink3",
        "pink3",
        "plum3",
        "violet",
        "gold3",
        "light_goldenrod3",
        "tan",
        "misty_rose3",
        "thistle3",
        "plum2",
        "yellow3",
        "khaki3",
        "light_yellow3",
        "grey84",
        "light_steel_blue1",
        "yellow2",
        "dark_olive_green1",
        "dark_sea_green1",
        "honeydew2",
        "light_cyan1",
        "red1",
        "deep_pink2",
        "deep_pink1",
        "magenta2",
        "magenta1",
        "orange_red1",
        "indian_red1",
        "hot_pink",
        "medium_orchid1",
        "dark_orange",
        "salmon1",
        "light_coral",
        "pale_violet_red1",
        "orchid2",
        "orchid1",
        "orange1",
        "sandy_brown",
        "light_salmon1",
        "light_pink1",
        "pink1",
        "plum1",
        "gold1",
        "light_goldenrod2",
        "navajo_white1",
        "misty_rose1",
        "thistle1",
        "yellow1",
        "light_goldenrod1",
        "khaki1",
        "wheat1",
        "cornsilk1",
        "grey100",
        "grey3",
        "grey7",
        "grey11",
        "grey15",
        "grey19",
        "grey23",
        "grey27",
        "grey30",
        "grey35",
        "grey39",
        "grey42",
        "grey46",
        "grey50",
        "grey54",
        "grey58",
        "grey62",
        "grey66",
        "grey70",
        "grey74",
        "grey78",
        "grey82",
        "grey85",
        "grey89",
        "grey93",
    ]

    STYLE_TEXT = Literal[
        "bold",
        "blink",
        "blink2",
        "conceal",
        "italic",
        "reverse",
        "strike",
        "underline",
        "underline2",
        "frame",
        "encircle",
        "overline",
    ]

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

        style: str | None

        if style_text is not None:
            parts.append(str(style_text))

        if foreground is not None:
            parts.append(str(foreground))

        if background is not None:
            parts.append("on " + str(background))

        style = " ".join(parts) if len(parts) > 0 else None

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

    HDPrint("ola", 1, 2, 3, 4).print(foreground="cyan")

    # python -W ignore -m src.backend.utils.heitor_print
