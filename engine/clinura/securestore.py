"""Encryption at rest for Clinura's SQLite databases.

The databases run in memory while Clinura is open and are written to disk only as
AES-256-GCM encrypted snapshots (atomic replace, every few seconds and on shutdown).
Nothing readable ever touches the disk.

Key, in order of preference:
1. ``CLINURA_DATA_KEY`` – a passphrase (stretched with scrypt) for headless setups;
2. the OS keychain (macOS Keychain / Windows Credential Manager / Secret Service);
3. a ``data/.clinura.key`` file readable only by the current user (with a warning).

Existing unencrypted databases are imported once and the plaintext files removed.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import sqlite3
import threading
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

MAGIC = b"CLNR1"
KEYRING_SERVICE = "clinura"
KEYRING_USER = "data-key"
SQLITE_HEADER = b"SQLite format 3\x00"


def _key_file(directory: Path) -> Path:
    return directory / ".clinura.key"


def load_key(directory: Path, create: bool = True) -> bytes:
    passphrase = os.environ.get("CLINURA_DATA_KEY")
    if passphrase:
        return hashlib.scrypt(
            passphrase.encode(), salt=b"clinura-data-v1", n=2**15, r=8, p=1, maxmem=2**26, dklen=32
        )
    try:
        import keyring

        stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
        if stored:
            return base64.b64decode(stored)
        if create:
            key = secrets.token_bytes(32)
            keyring.set_password(KEYRING_SERVICE, KEYRING_USER, base64.b64encode(key).decode())
            return key
    except Exception as exc:  # no keychain available
        logger.warning("OS keychain unavailable (%s); using a protected key file", exc)
    path = _key_file(directory)
    if path.exists():
        return base64.b64decode(path.read_bytes())
    if not create:
        raise RuntimeError("No data key found")
    key = secrets.token_bytes(32)
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(base64.b64encode(key))
    return key


def rotate_key(directory: Path) -> bytes:
    """Replace the data key (crypto-shredding: old snapshots become unreadable)."""
    if os.environ.get("CLINURA_DATA_KEY"):
        return load_key(directory)
    key = secrets.token_bytes(32)
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, base64.b64encode(key).decode())
        return key
    except Exception:
        path = _key_file(directory)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(base64.b64encode(key))
        return key


class EncryptedDatabase:
    """In-memory SQLite connection persisted as an encrypted file."""

    def __init__(self, path: Path, lock: threading.Lock, autosave_s: float = 10.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = lock
        self._key = load_key(self.path.parent)
        self._aad = b"clinura:" + self.path.name.encode()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self._load()
        self._saved_changes = self.conn.total_changes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._autosave, args=(autosave_s,), daemon=True)
        self._thread.start()

    def _load(self) -> None:
        if not self.path.exists():
            return
        blob = self.path.read_bytes()
        if blob.startswith(SQLITE_HEADER):
            # One-time migration of an old plaintext database.
            source = sqlite3.connect(str(self.path))
            source.backup(self.conn)
            source.close()
            self._write(self.conn.serialize())
            for suffix in ("-wal", "-shm"):
                Path(str(self.path) + suffix).unlink(missing_ok=True)
            logger.info("Encrypted existing database %s", self.path.name)
            return
        if not blob.startswith(MAGIC):
            raise RuntimeError(f"{self.path} is not a Clinura database")
        nonce = blob[len(MAGIC) : len(MAGIC) + 12]
        data = bytearray(AESGCM(self._key).decrypt(nonce, blob[len(MAGIC) + 12 :], self._aad))
        # An image copied from a WAL-mode database cannot be opened in memory: switch the
        # header's read/write format bytes back to the rollback journal.
        if len(data) > 19 and data[18] == 2:
            data[18] = data[19] = 1
        self.conn.deserialize(bytes(data))

    def _write(self, data: bytes) -> None:
        nonce = secrets.token_bytes(12)
        payload = MAGIC + nonce + AESGCM(self._key).encrypt(nonce, data, self._aad)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def save(self, force: bool = False) -> None:
        """Snapshot and encrypt. Caller may hold the store lock or not."""
        if not force and self.conn.total_changes == self._saved_changes:
            return
        data = self.conn.serialize()
        self._saved_changes = self.conn.total_changes
        self._write(data)

    def _autosave(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                with self._lock:
                    self.save()
            except Exception:
                logger.exception("Encrypted save failed")

    def rekey(self, key: bytes) -> None:
        self._key = key
        self.save(force=True)

    def close(self) -> None:
        self._stop.set()
        self.save(force=True)
        self.conn.close()
