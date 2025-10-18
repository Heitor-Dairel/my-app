from datetime import date
from typing import Literal


# python -m src.backend.modules.math.pascoa


class CalcYear:

    HOLIDAY = Literal["Carnival", "Good Friday", "Corpus Christi"] | None
    CALENDAR = Literal["Gregorian", "Julian"] | None

    def __init__(self, year: int) -> None:

        self.year: int = year

    def computus(
        self, calendar: CALENDAR = None, holiday_type: HOLIDAY = None
    ) -> date | None:

        if self.year <= 0:
            msg = "Invalid year, enter a year greater than 0"
            raise Exception(msg)

        if calendar is None:

            if self.year >= 1582:

                var_a: int = self.year % 19
                var_b: int = self.year // 100
                var_c: int = self.year % 100
                var_d: int = var_b // 4
                var_e: int = var_b % 4
                var_f: int = (var_b + 8) // 25
                var_g: int = (var_b - var_f + 1) // 3
                var_h: int = (19 * var_a + var_b - var_d - var_g + 15) % 30
                var_i: int = var_c // 4
                var_k: int = var_c % 4
                var_l: int = (32 + 2 * var_e + 2 * var_i - var_h - var_k) % 7
                var_m: int = (var_a + 11 * var_h + 22 * var_l) // 451
                var_month: int = (var_h + var_l - 7 * var_m + 114) // 31
                var_day: int = (var_h + var_l - 7 * var_m + 114) % 31 + 1

                return date(self.year, var_month, var_day)

            var_a: int


# .strftime("%d/%m/%Y")
val = CalcYear(0).computus()

print(val)
