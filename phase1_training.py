import torch
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():

    # --------------------------------------------------
    # Performance Optimizations
    # --------------------------------------------------
    torch.backends.cudnn.benchmark = True

    device = 0 if torch.cuda.is_available() else "cpu"

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU VRAM : {vram_gb:.2f} GB")


    # --------------------------------------------------
    # Data Augmentation
    # --------------------------------------------------

    aug_args = dict(
        mosaic=1.0,
        mixup=0.2,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
    )

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    model = YOLO("yolo26s.pt")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    model.train(

        data="data_phase1.yaml",

        epochs=50,

        imgsz=640,

        batch=8,

        device=device,

        optimizer="AdamW",

        lr0=1e-3,

        lrf=0.01,

        warmup_epochs=3,

        freeze=10,

        amp=True,

        cache="disk",

        workers=8,

        patience=15,

        save_period=10,

        project="runs/phase1",

        name="pretrain_v1",

        val=False,                 # Validation disabled

        verbose=True,

        **aug_args

    )

    # --------------------------------------------------
    # Validation after training
    # --------------------------------------------------

    best_model = YOLO(
        "runs/detect/runs/phase1/pretrain_v1/weights/best.pt"
    )

    metrics = best_model.val(

        data="data_phase1.yaml",

        batch=2,

        imgsz=640,

        workers=4

    )

    print(metrics)


if __name__ == "__main__":
    freeze_support()
    main()