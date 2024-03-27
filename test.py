from ultralytics import YOLO

print("Im Running")
model = YOLO('yolov8n.pt')

results = model.predict(source="0", show=True)

print(results)
