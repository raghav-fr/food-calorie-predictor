import os
import sys
import json
import time
import shutil
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — all 7 datasets with their slugs and targets
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path("data_folder")   # Change this to your preferred root

DATASETS = [
    # ── Phase 1: Classification datasets (pseudo-bbox) ──────────────────
    {
        "id":          "phase1_food101",
        "phase":       1,
        "slug":        "kmader/food41",
        "target_dir":  BASE_DIR / "datasets/phase1/food101",
        "label":       "Food-101",
        "expected_min_images": 90_000,
        "notes":       "101 global classes, ~101K images",
    },
    {
        "id":          "phase1_chinese",
        "phase":       1,
        "slug":        "yihfeng/chinesefoodnet",
        "target_dir":  BASE_DIR / "datasets/phase1/chinese",
        "label":       "ChineseFoodNet",
        "expected_min_images": 150_000,
        "notes":       "208 Chinese classes, ~185K images",
    },
    {
        "id":          "phase1_mafood",
        "phase":       1,
        "slug":        "theviz/mafood121",
        "target_dir":  BASE_DIR / "datasets/phase1/mafood",
        "label":       "MAFood-121",
        "expected_min_images": 18_000,
        "notes":       "121 multi-cuisine classes, ~21K images",
    },
    # ── Phase 2: Detection datasets (real bounding boxes) ────────────────
    {
        "id":          "phase2_uec256",
        "phase":       2,
        "slug":        "rkuo2000/uecfood256",
        "target_dir":  BASE_DIR / "datasets/phase2/uec256",
        "label":       "UEC-256",
        "expected_min_images": 25_000,
        "notes":       "256 Japanese classes, ~31K images",
    },
    {
        "id":          "phase2_foodrecog22",
        "phase":       2,
        "slug":        "sainikhileshreddy/food-recognition-2022",
        "target_dir":  BASE_DIR / "datasets/phase2/foodrecog2022",
        "label":       "Food Recog 2022",
        "expected_min_images": 38_000,
        "notes":       "498 classes, ~44K COCO-format images",
    },
    {
        "id":          "phase2_foodseg103",
        "phase":       2,
        "slug":        "ggrill/foodseg103",
        "target_dir":  BASE_DIR / "datasets/phase2/foodseg103",
        "label":       "FoodSeg103",
        "expected_min_images": 6_000,
        "notes":       "103 ingredients, ~7K pixel masks",
    },
    {
        "id":          "phase2_unimib",
        "phase":       2,
        "slug":        "dangvanthuc0209/unimib2016",
        "target_dir":  BASE_DIR / "datasets/phase2/unimib",
        "label":       "UNIMIB2016",
        "expected_min_images": 800,
        "notes":       "73 Italian tray classes, ~1K images",
    },
]

# Full project directory tree (created before downloads start)
ALL_DIRS = [
    BASE_DIR / "datasets/phase1/food101",
    BASE_DIR / "datasets/phase1/chinese",
    BASE_DIR / "datasets/phase1/mafood",
    BASE_DIR / "datasets/phase2/uec256",
    BASE_DIR / "datasets/phase2/foodrecog2022",
    BASE_DIR / "datasets/phase2/foodseg103",
    BASE_DIR / "datasets/phase2/unimib",
    BASE_DIR / "merged/phase1/images/train",
    BASE_DIR / "merged/phase1/images/val",
    BASE_DIR / "merged/phase1/labels/train",
    BASE_DIR / "merged/phase1/labels/val",
    BASE_DIR / "merged/phase2/images/train",
    BASE_DIR / "merged/phase2/images/val",
    BASE_DIR / "merged/phase2/labels/train",
    BASE_DIR / "merged/phase2/labels/val",
    BASE_DIR / "runs/phase1",
    BASE_DIR / "runs/phase2",
    BASE_DIR / "weights",
    BASE_DIR / "exports",
    BASE_DIR / "logs",
]

# Thread-safe status store
_lock   = threading.Lock()
_status = {}   # id -> dict with keys: state, start, end, error, image_count


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def log(msg: str, tag: str = "INFO"):
    ts  = datetime.now().strftime("%H:%M:%S")
    tag_str = f"[{tag}]".ljust(9)
    print(f"  {ts}  {tag_str}  {msg}", flush=True)


def set_status(ds_id: str, **kwargs):
    with _lock:
        _status.setdefault(ds_id, {}).update(kwargs)


def count_images(directory: Path) -> int:
    """Count .jpg and .png files recursively."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in exts)


def check_disk_space(required_gb: float = 150.0):
    total, used, free = shutil.disk_usage(str(BASE_DIR.parent))
    free_gb = free / 1e9
    print(f"\n  Disk space check:")
    print(f"    Free:     {free_gb:.1f} GB")
    print(f"    Required: {required_gb:.1f} GB (recommended)")
    if free_gb < required_gb:
        print(f"  ⚠  WARNING: Less than {required_gb} GB free."
              f" Downloads may fail or be incomplete.")
        answer = input("  Continue anyway? [y/N]: ").strip().lower()
        if answer != "y":
            print("  Aborted.")
            sys.exit(1)
    else:
        print(f"    ✓  Sufficient space available.")



def scaffold_directories():
    print("\n  Creating project directory tree...")
    created = 0
    for d in ALL_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created += 1
    total = len(ALL_DIRS)
    print(f"  ✓  {total} directories ready  ({created} newly created)")
    print(f"     Root: {BASE_DIR.resolve()}")


# ─────────────────────────────────────────────────────────────
# DOWNLOAD WORKER  (runs in its own thread per dataset)
# ─────────────────────────────────────────────────────────────

def download_dataset(ds: dict) -> dict:
    """
    Download one dataset using the kaggle CLI, unzip it in-place,
    then count images to verify the download.
    Returns a result dict.
    """
    ds_id      = ds["id"]
    slug       = ds["slug"]
    target     = Path(ds["target_dir"])
    label      = ds["label"]

    set_status(ds_id, state="starting", start=time.time())
    log(f"Starting  {label}  ({ds['notes']})", tag=label[:8])

    try:
        # ── Step 1: Download + unzip via kaggle CLI ──────────────────────
        cmd = [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", slug,
            "-p", str(target),
            "--unzip",
        ]

        log(f"Downloading {slug} → {target}", tag=label[:8])
        set_status(ds_id, state="downloading")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)
            # Only print progress-like lines to avoid spam
            if any(kw in line.lower() for kw in
                   ["downloading", "unzip", "extracting", "%", "done", "error", "warning"]):
                log(line, tag=label[:8])

        proc.wait()

        if proc.returncode != 0:
            error_msg = "\n".join(output_lines[-10:])
            raise RuntimeError(
                f"kaggle CLI exited with code {proc.returncode}.\n{error_msg}"
            )

        # ── Step 2: Count images to verify download ──────────────────────
        set_status(ds_id, state="verifying")
        log(f"Verifying image count...", tag=label[:8])

        img_count = count_images(target)
        expected  = ds["expected_min_images"]

        if img_count < expected:
            log(
                f"⚠  Only {img_count:,} images found "
                f"(expected ≥ {expected:,}). Download may be incomplete.",
                tag=label[:8],
            )
            state = "partial"
        else:
            state = "done"

        elapsed = time.time() - _status[ds_id]["start"]
        set_status(ds_id, state=state, end=time.time(), image_count=img_count)

        log(
            f"✓  {label} complete — {img_count:,} images "
            f"in {elapsed/60:.1f} min",
            tag=label[:8],
        )
        return {"id": ds_id, "label": label, "state": state,
                "image_count": img_count, "elapsed_s": elapsed}

    except Exception as e:
        elapsed = time.time() - _status[ds_id].get("start", time.time())
        set_status(ds_id, state="error", end=time.time(),
                   error=str(e), image_count=0)
        log(f"✗  {label} FAILED: {e}", tag="ERROR")
        return {"id": ds_id, "label": label, "state": "error",
                "image_count": 0, "elapsed_s": elapsed, "error": str(e)}


# ─────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────

def print_summary(results: list, total_elapsed: float):
    print("\n" + "─" * 60)
    print("  DOWNLOAD SUMMARY")
    print("─" * 60)

    total_images = 0
    errors       = []

    for r in sorted(results, key=lambda x: x["label"]):
        icon  = "✓" if r["state"] == "done" else \
                "~" if r["state"] == "partial" else "✗"
        mins  = r["elapsed_s"] / 60
        imgs  = r.get("image_count", 0)
        total_images += imgs

        line = (
            f"  {icon}  {r['label']:<20s}"
            f"  {imgs:>8,} images"
            f"  {mins:>5.1f} min"
        )
        if r["state"] == "error":
            errors.append(r)
            line += f"  ← ERROR"
        elif r["state"] == "partial":
            line += f"  ← PARTIAL"
        print(line)

    print("─" * 60)
    print(f"     Total images downloaded: {total_images:,}")
    print(f"     Total wall time:         {total_elapsed/60:.1f} min")

    if errors:
        print(f"\n  Failed datasets ({len(errors)}):")
        for e in errors:
            print(f"    • {e['label']}: {e.get('error','unknown error')}")
        print("\n  Re-run the script — it will skip already-downloaded datasets.")
    else:
        print("\n  All datasets downloaded successfully. ✓")

    # Save log to file
    log_data = {
        "timestamp":      datetime.now().isoformat(),
        "total_images":   total_images,
        "total_elapsed_s": total_elapsed,
        "results":        results,
    }
    log_path = BASE_DIR / "logs" / "download_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n  Log saved → {log_path}")
    print("─" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
# SKIP LOGIC — don't re-download if already complete
# ─────────────────────────────────────────────────────────────

def should_skip(ds: dict) -> bool:
    """Return True if the dataset looks already downloaded."""
    target = Path(ds["target_dir"])
    if not target.exists():
        return False
    img_count = count_images(target)
    if img_count >= ds["expected_min_images"]:
        log(
            f"Skipping {ds['label']} — already has {img_count:,} images",
            tag="SKIP",
        )
        return True
    return False


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  YOLO26s Food Pipeline — Dataset Downloader")
    print("  Downloads all 7 datasets in parallel")
    print("═" * 60)

    # ── Pre-flight checks ────────────────────────────────────
    check_disk_space(required_gb=150.0)
    scaffold_directories()

    # ── Filter out already-downloaded datasets ───────────────
    pending = [ds for ds in DATASETS if not should_skip(ds)]

    if not pending:
        print("\n  All datasets are already downloaded. Nothing to do.\n")
        return

    print(f"\n  Queued for download: {len(pending)} / {len(DATASETS)} datasets")
    for ds in pending:
        print(f"    • {ds['label']:<20s}  {ds['notes']}")

    # ── How many parallel workers? ───────────────────────────
    # Bandwidth is usually the bottleneck, not CPU.
    # 3 parallel downloads is a good balance for most home connections.
    # Change MAX_WORKERS to 1 for a strict sequential download.
    MAX_WORKERS = min(3, len(pending))
    print(f"\n  Parallel workers: {MAX_WORKERS}")
    print(f"  Starting at: {datetime.now().strftime('%H:%M:%S')}")
    print("─" * 60)

    wall_start = time.time()
    results    = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_dataset, ds): ds for ds in pending}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Add skipped datasets to results list for summary
    for ds in DATASETS:
        if ds not in pending:
            img_count = count_images(Path(ds["target_dir"]))
            results.append({
                "id": ds["id"], "label": ds["label"],
                "state": "skipped", "image_count": img_count,
                "elapsed_s": 0,
            })

    total_elapsed = time.time() - wall_start
    print_summary(results, total_elapsed)


if __name__ == "__main__":
    main()