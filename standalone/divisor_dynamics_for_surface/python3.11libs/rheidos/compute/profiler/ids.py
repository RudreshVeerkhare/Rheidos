from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import threading
from typing import Dict, Optional


def stable_hash_int(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "big", signed=False)


@dataclass
class StableIdMap:
    _name_to_id: Dict[str, int] = field(default_factory=dict)
    _id_to_name: Dict[int, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def intern(self, name: str) -> int:
        existing = self._name_to_id.get(name)
        if existing is not None:
            return existing
        new_id = stable_hash_int(name)
        with self._lock:
            existing = self._name_to_id.get(name)
            if existing is not None:
                return existing
            self._name_to_id[name] = new_id
            self._id_to_name.setdefault(new_id, name)
        return new_id

    def name_for(self, value: int) -> Optional[str]:
        return self._id_to_name.get(value)

    def all_ids(self) -> Dict[int, str]:
        with self._lock:
            return dict(self._id_to_name)


PRODUCER_IDS = StableIdMap()
RESOURCE_IDS = StableIdMap()
