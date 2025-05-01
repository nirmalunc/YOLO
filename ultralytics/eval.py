from ultralytics import YOLO


def main():
    # Load your trained model
    #model = YOLO('runs/detect/train26/weights/best.pt') #BiFPN
    #model = YOLO('yolov8n.pt')
    model = YOLO('runs/detect/train34/weights/best.pt') #BiFPN & SimAM

    # Evaluate on COCO val2017 using built-in coco.yaml
    metrics = model.val(data='coco.yaml', split='val', iou=1.0)  # or specify full path to your custom YAML

    # print(metrics.box.map)      # mAP@0.5:0.95
    # print(metrics.box.map50)    # mAP@0.5
    # print(metrics.box.map75)    # mAP@0.75
    # print(metrics.box.maps)     # per-class AP


if __name__ == "__main__":
    main()