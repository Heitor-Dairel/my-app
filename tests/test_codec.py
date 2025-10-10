from backend.modules.codec import EncryptionCodeDecode
from backend.modules.math.fibonacci import fibonacci_serie

# import timeit

# tempo = timeit.timeit(
#     "fibonacci_serie(1000, single=True)",
#     setup="from __main__ import fibonacci_serie",
#     number=1000,
# )
# print(
#     f"fibonacci_serie(1000, single=True)\
#         = Executou 1000 vezes em {tempo} segundos"
# )

# print(fibonacci_serie(4, single=False))
# print(fibonacci_serie(10, single=True))


def test_fibonacci():
    assert fibonacci_serie(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_acento():
    assert (
        EncryptionCodeDecode(
            "wmKWlws3NI7S3N9R8+izGwZtX5fz/IxqzgaPy7V7LR3Z+54LOcq"
            "xYdxwRNBJzyj5Y/Pn6FFtgN5H17HvOz9X7dst"
            "B1QUCw==",
            "oio",
        ).decrypt
        == "fdfsfs"
    )


# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# python -m tests.test_codec
