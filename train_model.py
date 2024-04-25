from ultralytics import YOLO

# Load a model
model = YOLO("runs\\detect\\train\\weights\\best.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='SKU-110K.yaml', epochs=100, imgsz=640)

results = model.predict(source="0", show=True)

print(results)
