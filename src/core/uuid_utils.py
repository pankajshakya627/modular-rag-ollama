"""UUID generation utilities using UUID v7 for Modular RAG.

UUID v7 Benefits over UUID v4:
- Time-sortable: Embeds 48-bit Unix timestamp in most significant bits
- Better index locality: Sequential nature reduces B-tree fragmentation
- Faster inserts: ~33% faster in PostgreSQL benchmarks
- Natural ordering: No need for separate created_at columns
- Distributed-safe: Still globally unique without coordination

Note: This module avoids circular imports by not being automatically
loaded via src/__init__.py. Import directly:
    from src.utils.uuid_utils import generate_uuid
"""
import uuid
from typing import Union

# Try to import uuid7 (time-sortable UUIDs)
_uuid7_func = None

try:
    # Python 3.14+ has native uuid.uuid7() support
    _uuid7_func = getattr(uuid, 'uuid7', None)
except AttributeError:
    pass

if _uuid7_func is None:
    try:
        # The uuid7 package provides uuid_extensions module
        from uuid_extensions import uuid7 as _uuid7_func
    except ImportError:
        try:
            # Alternative: uuid_utils package (Rust-based, faster)
            from uuid_utils import uuid7 as _uuid7_func
        except ImportError:
            _uuid7_func = None


def generate_uuid() -> str:
    """Generate a UUID v7 (time-sorted) or fallback to UUID v4."""
    if _uuid7_func is not None:
        return str(_uuid7_func())
    # Fallback to uuid4
    return str(uuid.uuid4())


def generate_uuid_v4() -> str:
    """Generate a UUID v4 (random) - for cases where time-ordering is not needed."""
    return str(uuid.uuid4())


def parse_uuid(uuid_str: str) -> uuid.UUID:
    """Parse a UUID string into a UUID object."""
    return uuid.UUID(uuid_str)


def extract_timestamp_from_uuid7(uuid_str: str) -> int:
    """Extract the Unix timestamp (milliseconds) from a UUID v7.
    
    UUID v7 format: TTTTTTTT-TTTT-7XXX-XXXX-XXXXXXXXXXXX
    First 48 bits (12 hex chars) are the timestamp.
    """
    try:
        # Remove hyphens and get first 12 hex chars
        hex_str = uuid_str.replace("-", "")[:12]
        timestamp_ms = int(hex_str, 16)
        return timestamp_ms
    except (ValueError, IndexError):
        return 0


def is_uuid_v7(uuid_str: str) -> bool:
    """Check if a UUID string is version 7."""
    try:
        u = uuid.UUID(uuid_str)
        return u.version == 7
    except ValueError:
        return False


# Convenience aliases
uuid_v7 = generate_uuid
uuid_v4 = generate_uuid_v4
