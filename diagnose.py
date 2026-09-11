import torch
from pathlib import Path

def inspect_ckpt(path):
    print(f"\n{'='*60}")
    print(f"Checkpoint: {path}")
    print(f"{'='*60}")
    if not Path(path).exists():
        print("  FILE NOT FOUND")
        return None

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    epoch        = ckpt.get("epoch", "N/A")
    best_fitness = ckpt.get("best_fitness", "N/A")
    model_val    = ckpt.get("model")
    ema_val      = ckpt.get("ema")

    print(f"  epoch        : {epoch}")
    print(f"  best_fitness : {best_fitness}")
    print(f"  model type   : {type(model_val).__name__}")
    print(f"  ema type     : {type(ema_val).__name__}")

    # Check model
    if model_val is not None:
        try:
            nan_m = sum(1 for _, p in model_val.named_parameters() if torch.isnan(p).any())
            inf_m = sum(1 for _, p in model_val.named_parameters() if torch.isinf(p).any())
            nc    = getattr(list(model_val.model.children())[-1], "nc", "?")
            print(f"  model nc     : {nc}")
            print(f"  model NaN params: {nan_m}")
            print(f"  model Inf params: {inf_m}")
        except Exception as e:
            print(f"  model inspect error: {e}")
    else:
        print("  model        : *** NONE — CORRUPT ***")

    # Check ema (this is the Exponential Moving Average copy — often the 'best' weights)
    if ema_val is not None:
        try:
            nan_e = sum(1 for _, p in ema_val.named_parameters() if torch.isnan(p).any())
            inf_e = sum(1 for _, p in ema_val.named_parameters() if torch.isinf(p).any())
            nc_e  = getattr(list(ema_val.model.children())[-1], "nc", "?")
            print(f"  ema nc       : {nc_e}")
            print(f"  ema NaN params: {nan_e}")
            print(f"  ema Inf params: {inf_e}")
            print(f"  ema is usable : {nan_e == 0 and inf_e == 0}")
        except Exception as e:
            print(f"  ema inspect error: {e}")
    else:
        print("  ema          : NONE")

    return ckpt

# Check all relevant checkpoints
checkpoints = [
    "runs/detect/runs/phase1/pretrain_v1-2/weights/best.pt",
    "runs/detect/runs/phase1/pretrain_v1-2/weights/last.pt",
    "runs/detect/runs/phase1/pretrain_v1-2/weights/epoch40.pt",
    "runs/detect/runs/phase1/pretrain_v1-2/weights/epoch30.pt",
    "runs/detect/runs/phase1/pretrain_v1/weights/best.pt",
    "runs/detect/runs/phase1/pretrain_v1/weights/last.pt",
]

usable = []
for p in checkpoints:
    ckpt = inspect_ckpt(p)
    if ckpt is None:
        continue
    model = ckpt.get("model")
    ema   = ckpt.get("ema")
    if model is not None:
        try:
            nan_m = sum(1 for _, p2 in model.named_parameters() if torch.isnan(p2).any())
            if nan_m == 0:
                usable.append((p, "model", ckpt.get("epoch")))
        except:
            pass
    if ema is not None:
        try:
            nan_e = sum(1 for _, p2 in ema.named_parameters() if torch.isnan(p2).any())
            if nan_e == 0:
                usable.append((p, "ema", ckpt.get("epoch")))
        except:
            pass

print("\n" + "="*60)
print("USABLE CHECKPOINTS (no NaN weights):")
for path, source, epoch in usable:
    print(f"  [{source}] epoch={epoch}  {path}")
print("="*60)
