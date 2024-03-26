from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2

print("Im Running")
model = YOLO("yolov8x.pt")

results = model.predict(source="0", show=True)

print(results)
