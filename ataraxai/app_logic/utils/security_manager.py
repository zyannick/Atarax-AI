import os
from types import TracebackType
from typing import Optional, Type
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag


class SecurityManager:

    VAULT_CHECK_PLAINTEXT = b"VAULT_OK"

    def __init__(self, salt_path: str, check_path: str):
        """
        Initializes the SecurityManager instance with the specified salt and check file paths.

        Args:
            salt_path (str): The file path where the salt is stored or will be created.
            check_path (str): The file path used for verification or checking purposes.

        Attributes:
            salt_path (str): Path to the salt file.
            check_path (str): Path to the check file.
            salt (bytes): The loaded or newly created salt value.
            key (Any): Placeholder for the cryptographic key, initialized as None.
        """
        self.salt_path = salt_path
        self.check_path = check_path
        self.salt = self._load_or_create_salt()
        self.key = None

    def _load_or_create_salt(self) -> bytes:
        """
        Loads a cryptographic salt from the specified file path if it exists; otherwise, generates a new random salt,
        saves it to the file, and returns it.

        Returns:
            bytes: The loaded or newly generated salt.
        """
        if os.path.exists(self.salt_path):
            with open(self.salt_path, "rb") as f:
                return f.read()
        else:
            salt = os.urandom(16)
            with open(self.salt_path, "wb") as f:
                f.write(salt)
            return salt

    def create_vault_check(self):
        """
        Creates a vault check file by encrypting a predefined plaintext and writing it to disk.

        Raises:
            RuntimeError: If the encryption key has not been derived.

        Side Effects:
            Writes the encrypted vault check to the file specified by `self.check_path`.
        """
        if not self.key:
            raise RuntimeError("Cannot create vault check: key not derived.")
        encrypted_check = self.encrypt(self.VAULT_CHECK_PLAINTEXT)
        with open(self.check_path, "wb") as f:
            f.write(encrypted_check)

    def verify_password(self) -> bool:
        """
        Verifies whether the provided key can successfully decrypt the vault check file.

        Returns:
            bool: True if the key is valid and the decrypted check file matches the expected plaintext,
            False otherwise. Raises FileNotFoundError if the check file does not exist.
        """
        if not self.key:
            return False
        if not os.path.exists(self.check_path):
            raise FileNotFoundError(
                "Vault check file not found. Vault may be uninitialized."
            )

        with open(self.check_path, "rb") as f:
            encrypted_check = f.read()

        try:
            decrypted_check = self.decrypt(encrypted_check)
            return decrypted_check == self.VAULT_CHECK_PLAINTEXT
        except InvalidTag:
            return False

    def derive_key(self, password: str) -> None:
        """
        Derives a cryptographic key from the provided password using PBKDF2-HMAC-SHA256.

        Args:
            password (str): The password to derive the key from.

        Raises:
            ValueError: If the password is empty.

        Side Effects:
            Sets the derived key to `self.key` using the instance's salt and stores it for later use.
        """
        if not password:
            raise ValueError("Password cannot be empty.")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=480000,  # Increased iterations for better security
            backend=default_backend(),
        )
        self.key = kdf.derive(password.encode())

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypts the given data using AES-GCM with the current encryption key.

        Args:
            data (bytes): The plaintext data to encrypt.

        Returns:
            bytes: The encrypted data, with the nonce prepended.

        Raises:
            RuntimeError: If the encryption key is not available (vault is locked).

        Note:
            The returned value consists of the 12-byte nonce followed by the ciphertext.
        """
        if not self.key:
            raise RuntimeError("Encryption key is not available. The vault is locked.")

        aesgcm = AESGCM(self.key)
        nonce = os.urandom(12)
        return nonce + aesgcm.encrypt(nonce, data, None)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypts the provided encrypted data using AES-GCM.

        Args:
            encrypted_data (bytes): The data to decrypt, expected to have the nonce as the first 12 bytes followed by the ciphertext.

        Returns:
            bytes: The decrypted plaintext.

        Raises:
            RuntimeError: If the decryption key is not available (vault is locked).
            cryptography.exceptions.InvalidTag: If decryption fails due to authentication error or corrupted data.
        """
        if not self.key:
            raise RuntimeError("Decryption key is not available. The vault is locked.")

        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]

        aesgcm = AESGCM(self.key)
        return aesgcm.decrypt(nonce, ciphertext, None)

    def lock(self) -> None:
        """
        Locks the security manager by clearing the current key.

        This method sets the `key` attribute to `None`, effectively disabling access
        until a new key is set.
        """
        self.key = None
        
    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.lock()
