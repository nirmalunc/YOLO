from ultralytics import YOLO

import torch
import os

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    model = YOLO('yolov8_simam.yaml')

    results = model.train(data="datasets/coco.yaml", epochs=40, imgsz=640, device=0, batch=32, workers=8)

if __name__ == "__main__":
    main()