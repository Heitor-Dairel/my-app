from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
from argon2.low_level import hash_secret_raw, Type
import os
import base64
import binascii


class EncryptionCodeDecode:
    r"""
    Encrypt and decrypt strings using AES-256-GCM with a password.

    This class allows converting plaintext strings into encrypted base64 strings
    and vice versa, using a secure key derived from a password and a random salt.
    The AES-256-GCM mode ensures confidentiality and integrity of the data.

    Attributes:
        value (str): The plaintext (for encryption) or base64 string (for decryption).
        password (str): The password used to derive the encryption key.

    Methods:
        encrypt -> str:
            Encrypt `value` and return a base64-encoded string containing salt, nonce, and ciphertext.
        decrypt -> str | None:
            Decrypt a base64-encoded string `value` using the password. Returns None if decryption fails.
    """

    def __init__(self, value: str, password: str) -> None:
        """
        Initialize the EncryptionCodeDecode instance.

        Args:
            value (str): The plaintext to encrypt or the base64 string to decrypt.
            password (str): Password used for key derivation.
        """

        self.value: str = value
        self.password: str = password

    @staticmethod
    def _generate_key(password: str, salt: bytes) -> bytes:
        r"""
        Derive a secure 256-bit key from a password and salt using Argon2id.

        This function generates a key suitable for AES-256 encryption by applying
        the Argon2id key derivation function with specified time cost, memory cost,
        and parallelism.

        Args:
            password (str): The password to derive the key from.
            salt (bytes): A unique salt to ensure uniqueness of the key.

        Returns:
            (bytes): 32-byte (256-bit) derived key.
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

    def encrypt(self) -> str:
        r"""
        Encrypt the instance's `value` using AES-256-GCM.

        Steps:
        1. Generate a random 32-byte salt and 16-byte nonce.
        2. Derive a secure key from the password and salt using `generate_key`.
        3. Encrypt the plaintext.
        4. Return a base64-encoded string combining salt + nonce + ciphertext.

        Returns:
            (str): Base64 string containing salt, nonce, and encrypted message.
        """

        salt: bytes = os.urandom(32)  # *larger salt (32 bytes)
        nonce: bytes = os.urandom(16)  # *larger nonce (16 bytes)
        key: bytes = EncryptionCodeDecode._generate_key(self.password, salt)
        aesgcm: AESGCM = AESGCM(key)
        encrypted: bytes = aesgcm.encrypt(nonce, self.value.encode(), None)
        # *We return salt + nonce + encrypted in base64
        return base64.b64encode(salt + nonce + encrypted).decode()

    def decrypt(self) -> str | None:
        r"""
        Decrypt the base64-encoded `value`.

        Steps:
        1. Decode the base64 string.
        2. Extract the salt (first 32 bytes), nonce (next 16 bytes), and ciphertext.
        3. Derive the key from the password and salt.
        4. Decrypt using AES-256-GCM.
        5. Return the plaintext if valid, otherwise None.

        Returns:
            (str | None): Decrypted plaintext, or None if decoding/authentication fails.
        """

        try:
            data: bytes = base64.b64decode(self.value)
            salt: bytes = data[:32]
            nonce: bytes = data[32:48]  # the next 16 bytes
            encrypted: bytes = data[48:]
            key: bytes = EncryptionCodeDecode._generate_key(self.password, salt)
            aesgcm: AESGCM = AESGCM(key)
            return aesgcm.decrypt(nonce, encrypted, None).decode()
        except (binascii.Error, InvalidTag):
            return None


if __name__ == "__main__":
    teste1 = EncryptionCodeDecode("fdfsfs", "oio").encrypt()
    print(
        EncryptionCodeDecode(
            "bdPzFEW1iTviXbnbzmzO1v8fCzWt/0w9UJ2Od6KZzgwU1cu1NC8VstuBygSNrhNL5+Vggu3x0eyDRu5CrMSubF+p0Q==",
            "h",
        ).decrypt()
    )
    print(
        EncryptionCodeDecode(
            "wmKWlws3NI7S3N9R8+izGwZtX5fz/IxqzgaPy7V7LR3Z+54LOcqxYdxwRNBJzyj5Y/Pn6FFtgN5H17HvOz9X7dstB1QUCw==",
            "oio",
        ).decrypt()
    )

    # python -W ignore -m src.backend.modules.codec.encryption_code_decode
