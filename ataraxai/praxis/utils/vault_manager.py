import os
from types import TracebackType
from typing import Optional, Type

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


from ataraxai.hegemonikon_py import SecureKey, SecureString, derive_and_protect_key  # type: ignore
from enum import Enum, auto
from dataclasses import dataclass


class VaultUnlockStatus(Enum):
    SUCCESS = auto()
    INVALID_PASSWORD = auto()
    ALREADY_UNLOCKED = auto()
    INCORRECT_PASSWORD = auto()
    UNINITIALIZED = auto()

class VaultInitializationStatus(Enum):
    SUCCESS = auto()
    ALREADY_INITIALIZED = auto()
    FAILED = auto()

@dataclass
class UnlockResult:
    status: VaultUnlockStatus
    error: Optional[str] = None


class VaultManager:
    VAULT_CHECK_PLAINTEXT = b"ATARAXAI_VAULT_OK"
    SALT_SIZE_BYTES = 16

    def __init__(self, salt_path: str, check_path: str):
        """
        Initializes the security manager with the specified salt and check file paths.

        Args:
            salt_path (str): The file path where the cryptographic salt is stored or will be created.
            check_path (str): The file path used for verification or integrity checks.

        Attributes:
            salt_path (str): Path to the salt file.
            check_path (str): Path to the check file.
            salt (bytes): The loaded or newly created cryptographic salt.
            _secure_key (Optional[SecureKey]): Cached secure key, initialized as None.
        """
        self.salt_path = salt_path
        self.check_path = check_path
        self.salt = self._load_or_create_salt()
        self._secure_key: Optional[SecureKey] = None

    def _load_or_create_salt(self) -> bytes:
        """
        Loads a cryptographic salt from the specified file path if it exists; otherwise, generates a new salt,
        saves it to the file, and returns it.

        Returns:
            bytes: The loaded or newly generated salt.

        Side Effects:
            May create a new file at self.salt_path if it does not exist.
        """
        if os.path.exists(self.salt_path):
            with open(self.salt_path, "rb") as f:
                return f.read()
        else:
            salt = os.urandom(self.SALT_SIZE_BYTES)
            with open(self.salt_path, "wb") as f:
                f.write(salt)
            return salt

    def unlock_vault(self, password: SecureString) -> UnlockResult:
        """
        Attempts to unlock the vault using the provided password.

        This method derives a temporary key from the given password and the instance's salt.
        It then verifies the derived key. If verification succeeds, the secure key is set and
        the vault is considered unlocked.

        Args:
            password (str): The password to unlock the vault.

        Returns:
            bool: True if the vault was successfully unlocked, False otherwise.
        """
        if not password:
            return UnlockResult(status=VaultUnlockStatus.INVALID_PASSWORD)

        try:
            temp_key = derive_and_protect_key(password=password, salt=self.salt)

            if self._verify_key(temp_key):
                self._secure_key = temp_key
                return UnlockResult(status=VaultUnlockStatus.SUCCESS)
            else:
                return UnlockResult(status=VaultUnlockStatus.INCORRECT_PASSWORD)
        except Exception as e:
            return UnlockResult(status=VaultUnlockStatus.UNINITIALIZED, error=str(e))

    def create_and_initialize_vault(self, password: SecureString):
        """
        Creates and initializes a secure vault using the provided password.

        This method derives a secure key from the given password and salt, then encrypts
        a predefined plaintext check value to verify future access. The encrypted check
        value is written to a file at the specified check path.

        Args:
            password (SecureString): The password to derive the secure key for the vault.

        Raises:
            ValueError: If the provided password is empty.
        """
        if not password:
            return VaultInitializationStatus.FAILED

        try:
            self._secure_key = derive_and_protect_key(password=password, salt=self.salt)
            encrypted_check = self.encrypt(self.VAULT_CHECK_PLAINTEXT)
            with open(self.check_path, "wb") as f:
                f.write(encrypted_check)
            return VaultInitializationStatus.SUCCESS
        except Exception:
            return VaultInitializationStatus.FAILED

    def _verify_key(self, key: SecureKey) -> bool:
        """
        Verifies the provided secure key by attempting to decrypt a vault check file.

        Args:
            key (SecureKey): The secure key to verify.

        Returns:
            bool: True if the key successfully decrypts the check file and matches the expected plaintext, False otherwise.

        Raises:
            FileNotFoundError: If the vault check file does not exist.
        """
        if not os.path.exists(self.check_path):
            raise FileNotFoundError(
                "Vault check file not found. Vault must be initialized first."
            )

        with open(self.check_path, "rb") as f:
            encrypted_check = f.read()

        try:
            aesgcm = AESGCM(key.data())
            nonce = encrypted_check[:12]
            ciphertext = encrypted_check[12:]
            decrypted_check = aesgcm.decrypt(nonce, ciphertext, None)
            return decrypted_check == self.VAULT_CHECK_PLAINTEXT
        except InvalidTag:
            return False

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypts the given data using AES-GCM with a securely stored key.

        Args:
            data (bytes): The plaintext data to encrypt.

        Returns:
            bytes: The encrypted data, consisting of a 12-byte nonce followed by the ciphertext.

        Raises:
            RuntimeError: If the secure key is not available (vault is locked).
        """
        if not self._secure_key:
            raise RuntimeError("Vault is locked. Cannot encrypt.")

        aesgcm = AESGCM(self._secure_key.data())
        nonce = os.urandom(12)
        return nonce + aesgcm.encrypt(nonce, data, None)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypts the provided encrypted data using AES-GCM with the stored secure key.

        Args:
            encrypted_data (bytes): The data to decrypt, expected to have the nonce as the first 12 bytes followed by the ciphertext.

        Returns:
            bytes: The decrypted plaintext data.

        Raises:
            RuntimeError: If the vault is locked and the secure key is not available.
        """
        if not self._secure_key:
            raise RuntimeError("Vault is locked. Cannot decrypt.")

        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]

        aesgcm = AESGCM(self._secure_key.data())
        return aesgcm.decrypt(nonce, ciphertext, None)

    def lock(self) -> None:
        self._secure_key = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.lock()


if __name__ == "__main__":
    vault_manager = VaultManager("vault.salt", "vault.check")
    password = SecureString("my_secure_password".encode("utf-8"))

    vault_manager.create_and_initialize_vault(password)

    unlock_result = vault_manager.unlock_vault(password)
    if unlock_result.status == VaultUnlockStatus.SUCCESS:
        print("Vault unlocked successfully.")
    else:
        print(f"Failed to unlock vault: {unlock_result.status.name}")

    wrong_password = SecureString("wrong_password".encode("utf-8"))
    unlock_result = vault_manager.unlock_vault(wrong_password)
    if unlock_result.status == VaultUnlockStatus.SUCCESS:
        print("Vault unlocked successfully with wrong password, which is unexpected.")
    else:
        print(f"Failed to unlock vault with wrong password: {unlock_result.status.name}")