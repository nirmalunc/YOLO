from ultralytics import YOLO

model = YOLO('yolov8_bifpn.yaml')

results = model.train(data="datasets/coco128.yaml", epochs=100, imgsz=640, device="mps")