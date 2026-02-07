import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

# Импортируем созданные ранее модули базы данных
from . import models as db_models
from . import crud, database
from .database import engine, get_db

# --- КОНФИГУРАЦИЯ ПУТЕЙ ---
# Т.к. файл теперь в папке Backend, BASE_DIR — это корень проекта
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "Model", "my_plant_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "Model", "classes.txt")
UPLOAD_DIR = os.path.join(CURRENT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Глобальные переменные для ML
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
    print("Инициализация системы...")
    
    # 1. Создаем таблицы в БД (если нет)
    db_models.Base.metadata.create_all(bind=engine)
    
    # 2. Загружаем классы
    if not os.path.exists(CLASSES_PATH):
        print(f"Критическая ошибка: не найден {CLASSES_PATH}")
    else:
        with open(CLASSES_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    
    # 3. Загружаем модель
    if os.path.exists(MODEL_PATH):
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Модель успешно загружена!")
    else:
        print(f"Предупреждение: файл модели не найден по пути {MODEL_PATH}")

    yield
    # Тут можно добавить логику закрытия ресурсов при выключении

app = FastAPI(title="Plant Disease Classifier", lifespan=lifespan)

# CORS для работы фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    # Ищем index.html в папке Frontend (на уровень выше Backend)
    frontend_path = os.path.join(BASE_DIR, "Frontend", "index.html")
    try:
        with open(frontend_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл index.html не найден")

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Разрешены только JPEG и PNG")

    try:
        # Читаем изображение
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Инференс модели
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = F.softmax(output[0], dim=0)
            
        conf, index = torch.max(probabilities, 0)
        
        label = class_names[index.item()]
        score = round(conf.item() * 100, 2)

        # --- СОХРАНЕНИЕ В БАЗУ ДАННЫХ ---
        # Опционально: сохраняем файл физически
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)

        crud.create_diagnosis(
            db=db,
            filename=file.filename,
            disease=label,
            confidence=score
        )
        # -------------------------------

        return {
            "disease": label,
            "confidence": f"{score}%",
            "all_predictions": {
                class_names[i]: f"{prob.item() * 100:.2f}%" 
                for i, prob in enumerate(probabilities) if prob > 0.01
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@app.get("/history")
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    return crud.get_diagnoses(db, limit=limit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)