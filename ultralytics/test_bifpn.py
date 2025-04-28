from ultralytics import YOLO

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

model = YOLO('yolov8_bifpn.yaml')

results = model.train(data="datasets/coco.yaml", epochs=25, imgsz=640, device=0, batch=0.9, workers=16)