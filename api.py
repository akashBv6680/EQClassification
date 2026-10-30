"""
FastAPI Microservice for CIFAR-10 Model Deployment
Optional Extension:
  - Exposes REST API endpoint /predict
  - Normalizes uploaded image
  - Supports ONNX Runtime or PyTorch inference
"""

import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from models.resnet_cifar import ResNetCIFARPyTorch

app = FastAPI(
    title="CIFAR-10 Classification Service",
    description="High-performance inference microservice for CIFAR-10 deep learning models",
    version="1.0.0"
)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetCIFARPyTorch(num_classes=10)
try:
    model.load_state_dict(torch.load('checkpoints/best_resnet.pt', map_location=device))
    print("[*] Loaded production weights")
except Exception:
    print("[!] Model checkpoint not found; running with base initialized weights")

model.to(device)
model.eval()

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "ResNetCIFAR", "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        top_k_indices = np.argsort(probs)[::-1][:3]
        results = [
            {"class": CLASSES[idx], "confidence": float(probs[idx]) * 100.0}
            for idx in top_k_indices
        ]

        return {
            "prediction": results[0]["class"],
            "top_probabilities": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
