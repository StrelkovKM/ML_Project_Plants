from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Путь к файлу базы данных в корне проекта
SQLALCHEMY_DATABASE_URL = "sqlite:///./plants.db"

# Создаем "движок" для SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Фабрика сессий для работы с БД
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс, от которого будут наследоваться наши модели (таблицы)
Base = declarative_base()

# Зависимость для получения сессии в эндпоинтах FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()