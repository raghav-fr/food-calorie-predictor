from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/phase1/pretrain_v1-2/weights/best.pt"
)

model.val(
    data="data_phase1.yaml",
    imgsz=640,
    batch=4,
    workers=0,
)