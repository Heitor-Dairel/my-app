# from typing import Any, Literal, Type, TypeVar
# from pydantic import BaseModel, TypeAdapter

# # from src.backend.utils import custom_print
# from pprint import pprint

# T = TypeVar("T", bound=BaseModel)


# class Nome(BaseModel):

#     ID_NOMES: int
#     NOMES: str


# class Pessoa(BaseModel):
#     TESTE: int
#     TESTE1: int
#     TESTe3: int
#     NOMES_LISTA: list[Nome]


# test: list[tuple[Any, ...]] = [
#     ("ID", "<DbType DB_TYPE_NUMBER>", 127, None, 0, -127, True),
#     ("TESTE", "<DbType DB_TYPE_NUMBER>", 127, None, 0, -127, True),
#     ("TESTE1", "<DbType DB_TYPE_NUMBER>", 127, None, 0, -127, True),
#     ("NOMES_LISTA_ID_NOMES", "<DbType DB_TYPE_NUMBER>", 127, None, 0, -127, True),
#     ("NOMES_LISTA_NOMES", "<DbType DB_TYPE_CHAR>", 6, 6, None, None, True),
# ]


# teste0 = [
#     (1, 123, 1233, 1, "OUTROS"),
#     (2, 123, 1233, 2, "OUTROS"),
#     (3, 123, 1233, 3, "HEITOR"),
#     (1, 123, 1233, 1, "OUTROS"),
#     (4, 123, 1233, None, None),
# ]

# teste1 = [(123, 123, 4, 5)]

# teste2 = (123, 123, 3)


# # for i in teste0:

# #     print(i)

# DICT_STYLE: dict[str, Any] = {
#     "upper": lambda a: a.upper(),
#     "lower": lambda a: a.lower(),
#     "title": lambda a: a,
# }


# def recurrence(
#     headers: list[tuple[Any, ...]],
#     results: tuple[Any, ...] | list[tuple[Any, ...]],
#     headers_style: Literal["upper", "lower", "title"] = "title",
# ) -> list[dict[str, Any]]:

#     headers_names: list[str] = [DICT_STYLE[headers_style](h[0]) for h in headers]

#     rows_result: list[tuple[Any, ...]] = (
#         [results] if isinstance(results, tuple) else results
#     )

#     result_query: list[dict[str, Any]] = [
#         dict(zip(headers_names, i)) for i in rows_result
#     ]

#     return result_query


# response = recurrence(test, teste0)

# # python -W ignore -m src.backend.core.CONFIG_TESTE

# pprint(response)
