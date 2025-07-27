import os
import tempfile
import pytest
from ataraxai.praxis.utils.vault_manager import VaultManager

class DummySecureKey:
    def __init__(self, key_bytes):
        self._key_bytes = key_bytes
    def data(self):
        return self._key_bytes

@pytest.fixture(autouse=True)
def patch_hegemonikon(monkeypatch):
    monkeypatch.setattr(
        "ataraxai.praxis.utils.vault_manager.derive_and_protect_key",
        lambda password, salt: DummySecureKey(b"0" * 32)
    )
    monkeypatch.setattr(
        "ataraxai.praxis.utils.vault_manager.SecureKey",
        DummySecureKey
    )
    monkeypatch.setattr(
        "ataraxai.praxis.utils.vault_manager.SecureString",
        str
    )

def test_create_and_initialize_vault_creates_check_file_and_encrypts(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        salt_path = os.path.join(tmpdir, "salt.bin")
        check_path = os.path.join(tmpdir, "check.bin")
        vm = VaultManager(salt_path, check_path)
        password = "testpassword"

        encrypted_data = b"nonce" + b"ciphertext"
        called = {}
        def fake_encrypt(data):
            called['data'] = data
            return encrypted_data
        vm.encrypt = fake_encrypt

        vm.create_and_initialize_vault(password)

        assert called['data'] == vm.VAULT_CHECK_PLAINTEXT

        assert os.path.exists(check_path)
        with open(check_path, "rb") as f:
            file_contents = f.read()
        assert file_contents == encrypted_data

def test_create_and_initialize_vault_raises_on_empty_password():
    with tempfile.TemporaryDirectory() as tmpdir:
        salt_path = os.path.join(tmpdir, "salt.bin")
        check_path = os.path.join(tmpdir, "check.bin")
        vm = VaultManager(salt_path, check_path)
        with pytest.raises(ValueError, match="Password cannot be empty"):
            vm.create_and_initialize_vault("")

def test_create_and_initialize_vault_sets_secure_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        salt_path = os.path.join(tmpdir, "salt.bin")
        check_path = os.path.join(tmpdir, "check.bin")
        vm = VaultManager(salt_path, check_path)
        password = "testpassword"

        vm.encrypt = lambda data: b"dummy"
        assert vm._secure_key is None
        vm.create_and_initialize_vault(password)
        assert isinstance(vm._secure_key, DummySecureKey)