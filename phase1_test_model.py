from ultralytics import YOLO


model = YOLO('runs/detect/runs/phase1/pretrain_v1-2/weights/best.pt')


# Visual check on one image
results = model('test_images/grilled-abalone-Korean-jeonbok-gui.jpg')
results[0].show()