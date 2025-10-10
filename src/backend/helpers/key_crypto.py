from argon2.low_level import hash_secret_raw, Type


def generate_key(password: str, salt: bytes) -> bytes:
    r"""
    Derive a secure 256-bit key from a password and salt using Argon2id.

    This function generates a key suitable for AES-256 encryption by applying
    the Argon2id key derivation function with specified time cost, memory cost,
    and parallelism.

    Args:
        password (str): The password to derive the key from.
        salt (bytes): A unique salt to ensure uniqueness of the key.

    Returns:
        return (bytes): 32-byte (256-bit) derived key.
    """

    key: bytes = hash_secret_raw(
        secret=password.encode(),
        salt=salt,
        time_cost=3,  # *number of iterations
        memory_cost=64 * 1024,  # *memory used in KB (64 MB)
        parallelism=2,  # *threads
        hash_len=32,  # *32 bytes = 256 bits for AES-256
        type=Type.ID,
    )
    return key
