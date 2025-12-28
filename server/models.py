import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, JSON
from .database import Base

class Incident(Base):
    __tablename__ = "incidents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    camera_id = Column(String, default="cam_01")
    threat_level = Column(String)  # CRITICAL, HIGH, WARN
    label = Column(String)         # SHOOTER, FIGHTING, etc.
    confidence = Column(Float)
    data = Column(JSON) 
