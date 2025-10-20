from datetime import date, timedelta
from typing import Literal, Final, Callable


class CalcYear:
    r"""
    Calculate Easter dates, movable holidays, and leap years for a given year.

    This class provides methods to:
        - Compute the date of Easter according to the Gregorian or Julian calendar.
        - Calculate the dates of Carnival, Good Friday, and Corpus Christi relative to Easter.
        - Determine if a year is a leap year.
        - Generate sequences of these dates across multiple years.

    Attributes:
        year (int): The base year used for calculations.
    """

    HOLIDAY = Literal["Carnival", "Good Friday", "Corpus Christi"] | None
    CALENDAR = Literal["Gregorian", "Julian"] | None
    SUM_HOLIDAY: Final[dict[str, Callable[[date], date]]] = {
        "Carnival": lambda x: x - timedelta(days=47),
        "Good Friday": lambda x: x - timedelta(days=2),
        "Corpus Christi": lambda x: x + timedelta(days=60),
    }

    ORDER_BY: Final[dict[str, bool]] = {"asc": False, "desc": True}

    def __init__(self, year: int) -> None:
        """
        Initialize the calculator with a specific year.

        Args:
            year (int): The base year for computations.
        """

        self.year: int = year

    @staticmethod
    def _easter_gregorian(year: int) -> date:
        r"""
        Calculate the date of Easter for the given year using the Gregorian calendar.

        Args:
            year (int): The year to calculate Easter for.

        Returns:
            date: The calculated Easter date.
        """

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
        r"""
        Calculate the date of Easter for the given year using the Julian calendar.

        Args:
            year (int): The year to calculate Easter for.

        Returns:
            date: The calculated Easter date.
        """

        var_a: int = year % 4
        var_b: int = year % 7
        var_c: int = year % 19
        var_d: int = (19 * var_c + 15) % 30
        var_e: int = (2 * var_a + 4 * var_b - var_d + 34) % 7
        var_month: int = (var_d + var_e + 114) // 31
        var_day: int = (var_d + var_e + 114) % 31 + 1

        return date(year, var_month, var_day)

    FLAG_CALENDAR: Final[dict[str, Callable[[int], date]]] = {
        "Gregorian": _easter_gregorian,
        "Julian": _easter_julian,
    }

    @staticmethod
    def _calendar_check(year: int, calendar_type: CALENDAR = None) -> date:
        r"""
        Return the Easter date for the given year and calendar type.

        Args:
            year (int): The year to calculate Easter for.
            calendar_type (CALENDAR, optional): "Gregorian" or "Julian". Defaults to None,
                in which case the Gregorian calendar is used for years >= 1582.

        Returns:
            date: The Easter date for the given year and calendar.
        """

        if calendar_type:
            return CalcYear.FLAG_CALENDAR[calendar_type](year)

        if year >= 1582:
            return CalcYear.FLAG_CALENDAR["Gregorian"](year)
        return CalcYear.FLAG_CALENDAR["Julian"](year)

    @staticmethod
    def _holiday_check(date: date, holiday_type: HOLIDAY = None) -> date:
        r"""
        Return the date of a movable holiday based on Easter.

        Args:
            date (date): The Easter date.
            holiday_type (HOLIDAY, optional): Type of holiday ("Carnival", "Good Friday", "Corpus Christi").
                Defaults to None, returning the input date.

        Returns:
            date: The holiday date.
        """

        if holiday_type:
            return CalcYear.SUM_HOLIDAY[holiday_type](date)
        return date

    @staticmethod
    def _leap_year_check(year: int) -> bool:
        r"""
        Determine if a year is a leap year.

        Args:
            year (int): The year to check.

        Returns:
            bool: True if the year is a leap year, False otherwise.
        """

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
        r"""
        Calculate Easter or a movable holiday date(s) for the current year.

        Args:
            calendar_type (CALENDAR, optional): "Gregorian" or "Julian". Defaults to None.
            holiday_type (HOLIDAY, optional): Type of holiday ("Carnival", "Good Friday", "Corpus Christi"). Defaults to None.
            all_results (bool, optional): If True, returns dates for multiple years. Defaults to False.
            range_years (int, optional): Number of years to calculate when all_results is True. Defaults to 10.
            year_step (int, optional): Step between years when all_results is True. Defaults to 1.
            order_by (Literal["asc", "desc"], optional): Sort order for multiple years. Defaults to "desc".

        Returns:
            (str | list[str] | None): Formatted date string(s). Returns None if the base year is invalid.

        Raises:
            Exception: If range_years <= 1.
        """

        if self.year <= 0:
            return None

        if range_years <= 1:
            msg = "Invalid range, enter a range greater than 1"
            raise Exception(msg)

        calc_year: int = self.year

        list_computus: list[str] = []

        result_computus: date = CalcYear._calendar_check(calc_year, calendar_type)

        if not all_results:

            return CalcYear._holiday_check(result_computus, holiday_type).strftime(
                "%d/%m/%Y"
            )

        for _ in range(range_years):

            result_computus: date = CalcYear._calendar_check(calc_year, calendar_type)

            list_computus.append(
                CalcYear._holiday_check(result_computus, holiday_type).strftime(
                    "%d/%m/%Y"
                )
            )

            calc_year += year_step

        return sorted(
            list_computus,
            key=lambda x: x.split("/")[-1],
            reverse=CalcYear.ORDER_BY[order_by],
        )

    def leap_year(
        self,
        *,
        all_results: bool = False,
        range_years: int = 10,
        year_step: int = 1,
        order_by: Literal["asc", "desc"] = "desc",
    ) -> dict[int, bool] | None:
        r"""
        Determine if the current year or a sequence of years are leap years.

        Args:
            all_results (bool, optional): If True, returns results for multiple years. Defaults to False.
            range_years (int, optional): Number of years to calculate when all_results is True. Defaults to 10.
            year_step (int, optional): Step between years when all_results is True. Defaults to 1.
            order_by (Literal["asc", "desc"], optional): Sort order for multiple years. Defaults to "desc".

        Returns:
            (dict[int, bool] | None): Dictionary mapping years to leap year status, or None if the base year is invalid.

        Raises:
            Exception: If range_years <= 1.
        """

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
                reverse=CalcYear.ORDER_BY[order_by],
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
