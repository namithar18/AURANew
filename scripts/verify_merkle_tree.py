# scripts/verify_merkle_tree.py
from aura.merkle_tree import MerkleTree, AuditEntry
from datetime import datetime
import os

# Clean start
test_path = 'audit_trail/test_merkle.json'
if os.path.exists(test_path):
    os.remove(test_path)

tree = MerkleTree(storage_path=test_path)

# Add 5 entries
for i in range(5):
    entry = AuditEntry(
        timestamp=datetime.utcnow().isoformat(),
        round_num=i,
        client_id=f'client_{i}',
        ae_update_norm=0.01 * i,
        head_update_norm=0.005 * i,
        local_val_accuracy=0.9 + 0.01 * i,
        ch1_trust_score=0.8,
        ch2_trust_score=0.7,
        classification='HEALTHY',
        aggregation_weight=0.25
    )
    tree.append(entry)

# Verify integrity
assert tree.verify_integrity(), "FAIL: integrity check failed on clean tree"
root_before = tree.root()
print(f"PASS: integrity verified. Root: {root_before[:16]}...")

# Tamper with an entry and verify detection
tree.entries[2]['ch1_trust_score'] = 0.0  # tamper
assert not tree.verify_integrity(), "FAIL: tamper not detected"
print("PASS: tampering correctly detected")

# Restore and verify recovery
tree.entries[2]['ch1_trust_score'] = 0.8
assert tree.verify_integrity(), "FAIL: integrity not restored after fix"
print("PASS: integrity restored after correction")

# Reload from disk and verify persistence
tree2 = MerkleTree(storage_path=test_path)
assert tree2.root() == root_before, "FAIL: root changed after reload"
assert len(tree2.entries) == 5, "FAIL: entries lost after reload"
print(f"PASS: persistence verified. {len(tree2.entries)} entries reloaded correctly")

os.remove(test_path)
print("All Merkle tree tests passed.")
