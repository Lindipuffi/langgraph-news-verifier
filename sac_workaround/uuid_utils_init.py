"""
Pure-Python drop-in replacement for uuid_utils/__init__.py.

WHY THIS EXISTS
---------------
Windows "Smart App Control" (enforced on this machine) blocks the unsigned
compiled extension `_uuid_utils.pyd` that the real `uuid_utils` package ships.
LangChain/LangSmith only need `uuid_utils.compat.uuid7`, and that layer only
needs a handful of functions from this top-level module. Everything here is
implemented in pure Python on top of the standard-library `uuid` module, so
no blocked binary is ever loaded.

This file is installed OVER `.venv/Lib/site-packages/uuid_utils/__init__.py`
by `apply_sac_workaround.py`. Re-run that script if you rebuild the venv.
"""

import secrets
import time
from uuid import (
    NAMESPACE_DNS,
    NAMESPACE_OID,
    NAMESPACE_URL,
    NAMESPACE_X500,
    RESERVED_FUTURE,
    RESERVED_MICROSOFT,
    RESERVED_NCS,
    RFC_4122,
    UUID,
    SafeUUID,
    getnode,
)
from uuid import uuid1 as _std_uuid1
from uuid import uuid3 as _std_uuid3
from uuid import uuid4 as _std_uuid4
from uuid import uuid5 as _std_uuid5

__version__ = "0.0.0-sac-shim"

NIL = UUID("00000000-0000-0000-0000-000000000000")
MAX = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")


def reseed_rng() -> None:
    """No-op: the stdlib `secrets` module reseeds itself; kept for API parity."""


# --- integer generators used by uuid_utils.compat -------------------------
def _uuid4_int() -> int:
    return _std_uuid4().int


def _uuid7_int(timestamp: int | None = None, nanos: int | None = None) -> int:
    """Return the 128-bit int of a UUIDv7 (RFC 9562, §5.7).

    Layout: 48-bit unix_ts_ms | version(0x7) | 12 rand | variant(0b10) | 62 rand
    """
    if timestamp is None:
        ns = time.time_ns()
    else:
        ns = timestamp * 1_000_000_000 + (nanos or 0)
    unix_ts_ms = (ns // 1_000_000) & 0xFFFFFFFFFFFF

    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    value = unix_ts_ms << 80
    value |= 0x7 << 76          # version
    value |= rand_a << 64
    value |= 0b10 << 62         # variant
    value |= rand_b
    return value


# --- UUID-returning helpers (mirror uuid_utils's public API) --------------
def uuid1(node=None, clock_seq=None) -> UUID:
    return _std_uuid1(node, clock_seq)


def uuid3(namespace, name) -> UUID:
    return _std_uuid3(namespace, name)


def uuid4() -> UUID:
    return _std_uuid4()


def uuid5(namespace, name) -> UUID:
    return _std_uuid5(namespace, name)


def uuid6(node=None, timestamp=None) -> UUID:
    """Minimal valid UUIDv6 (time-ordered). Not used by our stack, but present."""
    if timestamp is None:
        # 60-bit count of 100-ns intervals since the Gregorian epoch (1582-10-15).
        timestamp = (time.time_ns() // 100) + 0x01B21DD213814000
    time_high = (timestamp >> 28) & 0xFFFFFFFF
    time_mid = (timestamp >> 12) & 0xFFFF
    time_low = timestamp & 0x0FFF
    clock_seq = secrets.randbits(14)
    node = getnode() if node is None else node
    value = time_high << 96
    value |= time_mid << 80
    value |= 0x6 << 76          # version 6
    value |= time_low << 64
    value |= 0b10 << 62         # variant
    value |= clock_seq << 48
    value |= node & 0xFFFFFFFFFFFF
    return UUID(int=value)


def uuid7(timestamp=None, nanos=None) -> UUID:
    return UUID(int=_uuid7_int(timestamp, nanos))


def uuid8(bytes: bytes) -> UUID:
    """UUIDv8 built from 16 user-supplied bytes, with version/variant fixed."""
    if len(bytes) != 16:
        raise ValueError("uuid8 requires exactly 16 bytes")
    value = int.from_bytes(bytes, "big")
    value &= ~(0xF << 76)
    value |= 0x8 << 76          # version 8
    value &= ~(0b11 << 62)
    value |= 0b10 << 62         # variant
    return UUID(int=value)


__all__ = [
    "MAX",
    "NAMESPACE_DNS",
    "NAMESPACE_OID",
    "NAMESPACE_URL",
    "NAMESPACE_X500",
    "NIL",
    "RESERVED_FUTURE",
    "RESERVED_MICROSOFT",
    "RESERVED_NCS",
    "RFC_4122",
    "UUID",
    "SafeUUID",
    "__version__",
    "getnode",
    "reseed_rng",
    "uuid1",
    "uuid3",
    "uuid4",
    "uuid5",
    "uuid6",
    "uuid7",
    "uuid8",
    "_uuid4_int",
    "_uuid7_int",
]
