from sqlalchemy.orm import Session
from . import models

def create_diagnosis(db: Session, filename: str, disease: str, confidence: float):
    db_diagnosis = models.Diagnosis(
        filename=filename,
        disease_name=disease,
        confidence=confidence
    )

    db.add(db_diagnosis)
    db.commit()
    db.refresh(db_diagnosis)
    return db_diagnosis

def get_diagnoses(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Diagnosis).offset(skip).limit(limit).all()