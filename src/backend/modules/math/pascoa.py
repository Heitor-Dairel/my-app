from datetime import date, timedelta
from typing import Literal, Final, Callable


class CalcYear:

    HOLIDAY = Literal["Carnival", "Good Friday", "Corpus Christi"] | None
    CALENDAR = Literal["Gregorian", "Julian"] | None
    SUM_HOLIDAY: Final[dict[str, Callable[[date], date]]] = {
        "Carnival": lambda x: x - timedelta(days=47),
        "Good Friday": lambda x: x - timedelta(days=2),
        "Corpus Christi": lambda x: x + timedelta(days=60),
    }
    FLAG_CALENDAR: Final[dict[str, Callable[[int], date]]] = {
        "Gregorian": lambda x: CalcYear._easter_gregorian(x),
        "Julian": lambda x: CalcYear._easter_julian(x),
    }
    ORDER_BY: Final[dict[str, bool]] = {"asc": False, "desc": True}

    def __init__(self, year: int) -> None:

        self.year: int = year

    @staticmethod
    def _easter_gregorian(year: int) -> date:

        var_a: int = year % 19
        var_b: int = year // 100
        var_c: int = year % 100
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

        return date(year, var_month, var_day)

    @staticmethod
    def _easter_julian(year: int) -> date:

        var_a: int = year % 4
        var_b: int = year % 7
        var_c: int = year % 19
        var_d: int = (19 * var_c + 15) % 30
        var_e: int = (2 * var_a + 4 * var_b - var_d + 34) % 7
        var_month: int = (var_d + var_e + 114) // 31
        var_day: int = (var_d + var_e + 114) % 31 + 1

        return date(year, var_month, var_day)

    @staticmethod
    def _calendar_check(year: int, calendar_type: CALENDAR = None) -> date:

        if calendar_type:
            return CalcYear.FLAG_CALENDAR[calendar_type](year)

        if year >= 1582:
            return CalcYear.FLAG_CALENDAR["Gregorian"](year)
        return CalcYear.FLAG_CALENDAR["Julian"](year)

    @staticmethod
    def _holiday_check(date: date, holiday_type: HOLIDAY = None) -> date:

        if holiday_type:
            return CalcYear.SUM_HOLIDAY[holiday_type](date)
        return date

    @staticmethod
    def _leap_year_check(year: int) -> bool:

        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):

            return True

        return False

    def computus(
        self,
        *,
        calendar_type: CALENDAR = None,
        holiday_type: HOLIDAY = None,
        all_results: bool = False,
        range_years: int = 10,
        year_step: int = 1,
        order_by: Literal["asc", "desc"] = "desc",
    ) -> list[str] | str | None:

        if self.year <= 0:
            return None

        if range_years <= 1:
            msg = "Invalid range, enter a range greater than 1"
            raise Exception(msg)

        calc_year: int = self.year

        list_computus: list[str] = []

        result_computus: date = self._calendar_check(calc_year, calendar_type)

        if not all_results:

            return self._holiday_check(result_computus, holiday_type).strftime(
                "%d/%m/%Y"
            )

        for _ in range(range_years):

            result_computus: date = self._calendar_check(calc_year, calendar_type)

            list_computus.append(
                self._holiday_check(result_computus, holiday_type).strftime("%d/%m/%Y")
            )

            calc_year += year_step

        return sorted(
            list_computus,
            key=lambda x: x.split("/")[-1],
            reverse=self.ORDER_BY[order_by],
        )

    def leap_year(
        self,
        *,
        all_results: bool = False,
        range_years: int = 10,
        year_step: int = 1,
        order_by: Literal["asc", "desc"] = "desc",
    ) -> dict[int, bool] | None:

        if self.year <= 0:
            return None

        if range_years <= 1:
            msg = "Invalid range, enter a range greater than 1"
            raise Exception(msg)

        calc_year: int = self.year

        result_leap_year: dict[int, bool] = {}

        if not all_results:

            result_leap_year[calc_year] = CalcYear._leap_year_check(calc_year)

            return result_leap_year

        for _ in range(range_years):

            result_leap_year[calc_year] = CalcYear._leap_year_check(calc_year)

            calc_year += year_step

        return dict(
            sorted(
                result_leap_year.items(),
                key=lambda x: x[0],
                reverse=self.ORDER_BY[order_by],
            )
        )


if __name__ == "__main__":
    pascoa = CalcYear(1581).computus(
        calendar_type="Julian",
        holiday_type="Corpus Christi",
        all_results=True,
        range_years=2,
        year_step=1,
        order_by="asc",
    )

    bisx = CalcYear(2025).leap_year(
        all_results=True, range_years=15, year_step=-1, order_by="desc"
    )

    print(pascoa)
    print(bisx)
