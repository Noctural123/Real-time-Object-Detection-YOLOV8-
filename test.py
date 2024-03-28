from ultralytics import YOLO

# Replace model with path to the fine-tuned model trained on the
# 'coco128' dataset
model = YOLO("C:\\Users\\Lance\\Desktop\\YOLOV8\\runs\\detect\\train\\weights\\best.pt") 

# Use the fine-tuned model for real-time detection
results = model.predict(source=0, show=True)

print(results)
