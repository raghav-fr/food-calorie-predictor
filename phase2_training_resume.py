from xml.parsers.expat import model

import torch
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("runs/detect/runs/phase2/finetune_v1/weights/last.pt")
    model.train(resume=True)


if __name__ == "__main__":
    freeze_support()
    main()