"""
Repairs the Phase 1 checkpoint by copying the valid EMA weights
into the model slot (which is None due to an Ultralytics save bug).

Output: runs/detect/runs/phase1/pretrain_v1-2/weights/best_repaired.pt
"""
import torch
import copy
from pathlib import Path

SRC  = Path("runs/detect/runs/phase1/pretrain_v1-2/weights/best.pt")
DEST = Path("runs/detect/runs/phase1/pretrain_v1-2/weights/best_repaired.pt")

print(f"Loading: {SRC}")
ckpt = torch.load(SRC, map_location="cpu", weights_only=False)

print(f"  model type (before) : {type(ckpt['model']).__name__}")
print(f"  ema type            : {type(ckpt['ema']).__name__}")
print(f"  ema nc              : {list(ckpt['ema'].model.children())[-1].nc}")
print(f"  epoch               : {ckpt['epoch']}")

# ── Verify EMA is clean
ema = ckpt["ema"]
nan_count = sum(1 for _, p in ema.named_parameters() if torch.isnan(p).any())
inf_count = sum(1 for _, p in ema.named_parameters() if torch.isinf(p).any())
print(f"  ema NaN params      : {nan_count}")
print(f"  ema Inf params      : {inf_count}")

assert nan_count == 0, "EMA has NaN — cannot repair!"
assert inf_count == 0, "EMA has Inf — cannot repair!"

# ── Patch: set model = deep copy of ema
print("\nPatching: ckpt['model'] = copy of ckpt['ema'] ...")
ckpt["model"] = copy.deepcopy(ema)

# ── Set to eval mode and half precision (same as YOLO expects for inference/fine-tune loading)
ckpt["model"].eval()

# ── Verify patch
print(f"  model type (after)  : {type(ckpt['model']).__name__}")
nc_after = list(ckpt["model"].model.children())[-1].nc
print(f"  model nc (after)    : {nc_after}")
nan_after = sum(1 for _, p in ckpt["model"].named_parameters() if torch.isnan(p).any())
print(f"  model NaN params    : {nan_after}")

# ── Save repaired checkpoint
print(f"\nSaving repaired checkpoint to: {DEST}")
torch.save(ckpt, DEST)
size_mb = DEST.stat().st_size / 1e6
print(f"Saved — {size_mb:.1f} MB")
print("\nRepair COMPLETE. Use best_repaired.pt for Phase 2 training.")
