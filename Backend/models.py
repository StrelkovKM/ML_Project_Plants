from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone  # Добавляем timezone
from .database import Base

class Diagnosis(Base):
    __tablename__ = "diagnoses"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    disease_name = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))