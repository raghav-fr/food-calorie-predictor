import torch
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():

    torch.backends.cudnn.benchmark = True

    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {vram_gb:.2f} GB")
    else:
        print("Running on CPU")

    finetune_args = dict(
        epochs=120,
        imgsz=640,
        batch=8,                   # Change if OOM
        device=device,
        optimizer="AdamW",
        lr0=8e-5,
        lrf=0.001,
        cos_lr=True,
        warmup_epochs=8,
        freeze=10,

        cache=False,
        workers=6,

        amp=True,


        patience=10,
        save_period=5,

        project="runs/phase2",
        name="finetune_v1",
        seed=42,
        deterministic=False,
        multi_scale=False,
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
        cls=1.5,
        label_smoothing=0.05,
        val=False,
        verbose=True
    )

    model = YOLO(
        "runs/detect/runs/phase1/pretrain_v1-2/weights/best_repaired.pt"
    )

    print("\nStarting Phase-2 Fine-tuning...\n")

    model.train(

        data="data_phase2.yaml",

        **finetune_args

    )

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