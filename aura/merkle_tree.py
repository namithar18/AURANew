import hashlib
import json
from typing import List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class AuditEntry:
    """Single entry in the federated learning audit trail."""
    timestamp: str
    round_num: int
    client_id: str
    ae_update_norm: float
    head_update_norm: Optional[float]  # DC-FLTrust — may be None for single-channel
    local_val_accuracy: float
    ch1_trust_score: float
    ch2_trust_score: Optional[float]  # None if single-channel or warmup
    classification: str            # HEALTHY / UNDER_ATTACK / BYZANTINE / WARMUP
    aggregation_weight: float
    
    def to_dict(self):
        return asdict(self)

class MerkleTree:
    """
    Append-only Merkle tree for federated learning audit trail.
    Each leaf is a SHA-256 hash of an AuditEntry.
    Root hash changes whenever any entry changes — tamper-evident.
    Persisted to disk as a JSON file.
    """
    
    def __init__(self, storage_path: str = 'audit_trail/merkle_log.json'):
        self.storage_path = storage_path
        self.leaves: List[str] = []      # SHA-256 hashes of entries
        self.entries: List[dict] = []    # raw entries for readability
        self._load_or_init()
    
    def _hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _load_or_init(self):
        import os
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if os.path.exists(self.storage_path):
            with open(self.storage_path) as f:
                stored = json.load(f)
            self.leaves = stored['leaves']
            self.entries = stored['entries']
        
    def append(self, entry: AuditEntry):
        """Add a new entry. Fails loudly if storage write fails."""
        entry_json = json.dumps(entry.to_dict(), sort_keys=True)
        leaf_hash = self._hash(entry_json)
        
        # Chain with previous leaf for tamper-evidence
        if self.leaves:
            chained = self._hash(self.leaves[-1] + leaf_hash)
        else:
            chained = leaf_hash
        
        self.leaves.append(chained)
        self.entries.append(entry.to_dict())
        self._persist()
    
    def root(self) -> Optional[str]:
        """Compute current Merkle root. Returns None if tree is empty."""
        if not self.leaves:
            return None
        layer = self.leaves[:]
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])  # duplicate last if odd
            layer = [
                self._hash(layer[i] + layer[i+1])
                for i in range(0, len(layer), 2)
            ]
        return layer[0]
    
    def verify_integrity(self) -> bool:
        """Recompute all hashes and verify chain is unbroken."""
        if not self.entries:
            return True
        prev_hash = None
        for i, (entry, stored_hash) in enumerate(
            zip(self.entries, self.leaves)
        ):
            entry_json = json.dumps(entry, sort_keys=True)
            leaf_hash = self._hash(entry_json)
            if prev_hash:
                expected = self._hash(prev_hash + leaf_hash)
            else:
                expected = leaf_hash
            if expected != stored_hash:
                print(f"Integrity violation at entry {i}")
                return False
            prev_hash = stored_hash
        return True
    
    def _persist(self):
        """Persist to disk. Raises RuntimeError if write fails."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'leaves': self.leaves,
                    'entries': self.entries,
                    'root': self.root(),
                    'n_entries': len(self.entries)
                }, f, indent=2)
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Merkle tree write failed. "
                f"Audit trail integrity compromised. Error: {e}"
            )
