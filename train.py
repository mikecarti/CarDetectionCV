from ultralytics import YOLO
import torch
import os
# Load a model
if __name__ == "__main__":

    model = YOLO("yolo11m-obb.pt")  # load a pretrained model (recommended for training)

    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = model.train(data=os.path.join("datasets","YOLO_dataset_FINAL740", "data.yaml"), epochs=10000, batch=4,
                          imgsz=640, name='startup-nano', device=device,
                          patience=50)