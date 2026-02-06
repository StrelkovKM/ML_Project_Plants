import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# --- КОНФИГУРАЦИЯ ПУТЕЙ ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "Model", "my_plant_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "Model", "classes.txt")

# ОБЯЗАТЕЛЬНО ДОБАВИТЬ ЭТУ СТРОКУ:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Ищу модель по пути: {MODEL_PATH}")


# Глобальные переменные
model = None
class_names = []

# --- ПРЕПРОЦЕССИНГ ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names
    print("Загрузка классов и модели...")
    
    # 1. Загружаем классы
    with open(CLASSES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # 2. Инициализируем модель (ResNet18, как в Colab)
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    
    # Загружаем веса
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Модель готова! Загружено классов: {len(class_names)}")
    
    yield
    print("Отключение сервиса...")

# Инициализация FastAPI с lifespan
app = FastAPI(title="Plant Disease Classifier", lifespan=lifespan)

# Маршрут для фронтенда (главная страница)
@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open(os.path.join(CURRENT_DIR, "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл index.html не найден в папке FrontEnd")

# Маршрут для проверки состояния (переименован, чтобы не мешать фронтенду)
@app.get("/status")
def health_check():
    return {
        "status": "ok", 
        "classes_count": len(class_names), 
        "device": str(DEVICE),
        "model_loaded": model is not None
    }
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена на сервере")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Разрешены только форматы JPEG и PNG")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = F.softmax(output[0], dim=0)
            
        conf, index = torch.max(probabilities, 0)
        
        # Убрали [0], так как index — это уже скалярный тензор
        return {
            "disease": class_names[index.item()],
            "confidence": f"{conf.item() * 100:.2f}%",
            "all_predictions": {
                class_names[i]: f"{prob.item() * 100:.2f}%" 
                for i, prob in enumerate(probabilities) if prob > 0.01
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Запуск на порту 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)