import torch
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():

    # ==========================================================
    # Performance Optimizations
    # ==========================================================

    torch.backends.cudnn.benchmark = True

    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {vram_gb:.2f} GB")
    else:
        print("Running on CPU")

    # ==========================================================
    # Phase-2 Fine-tuning Configuration
    # ==========================================================

    finetune_args = dict(

        # -----------------------------
        # Training
        # -----------------------------
        epochs=120,
        imgsz=640,
        batch=8,                   # Change if OOM
        device=device,

        # -----------------------------
        # Optimizer
        # -----------------------------
        optimizer="AdamW",
        lr0=8e-5,
        lrf=0.001,
        cos_lr=True,
        warmup_epochs=8,

        # -----------------------------
        # Backbone
        # -----------------------------
        freeze=10,

        # -----------------------------
        # Data Loading
        # -----------------------------
        cache=False,
        workers=6,

        # -----------------------------
        # Mixed Precision
        # -----------------------------
        amp=True,

        # -----------------------------
        # Early Stopping
        # -----------------------------
        patience=10,

        # -----------------------------
        # Saving
        # -----------------------------
        save_period=5,

        # -----------------------------
        # Project
        # -----------------------------
        project="runs/phase2",
        name="finetune_v1",

        # -----------------------------
        # Reproducibility
        # -----------------------------
        seed=42,
        deterministic=False,

        # -----------------------------
        # Multi-scale
        # -----------------------------
        multi_scale=False,

        # -----------------------------
        # Augmentations
        # -----------------------------
        mosaic=0.3,
        close_mosaic=15,
        mixup=0.1,

        degrees=5,
        translate=0.10,
        scale=0.30,

        fliplr=0.5,
        flipud=0.05,

        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.30,

        # -----------------------------
        # Loss
        # -----------------------------
        cls=1.5,
        label_smoothing=0.05,

        # -----------------------------
        # Validation
        # -----------------------------
        val=False,

        verbose=True
    )

    # ==========================================================
    # Load Phase-1 Weights
    # ==========================================================

    model = YOLO(
        "runs/detect/runs/phase1/pretrain_v1-2/weights/best.pt"
    )

    print("\nStarting Phase-2 Fine-tuning...\n")

    # ==========================================================
    # Train
    # ==========================================================

    model.train(

        data="data_phase2.yaml",

        **finetune_args

    )

    # ==========================================================
    # Validation
    # ==========================================================

    print("\nLoading best model...\n")

    best_model = YOLO(
        "runs/detect/runs/phase2/finetune_v1/weights/best.pt"
    )

    metrics = best_model.val(

        data="data_phase2.yaml",

        imgsz=640,

        batch=2,

        workers=4

    )

    print("\n==============================")
    print("Final Validation Results")
    print("==============================")

    print(f"mAP@50      : {metrics.box.map50:.4f}")
    print(f"mAP@50-95   : {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")

    print("\nBest model saved at:")
    print("runs/detect/runs/phase2/finetune_v1/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    main()