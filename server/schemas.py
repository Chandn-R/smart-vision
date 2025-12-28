from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# --- Nested Models matching the user's requested JSON structure ---
class Detection(BaseModel):
    class_: str = Field(..., alias="class")
    confidence: float
    bbox: List[int]

class ActionClassification(BaseModel):
    class_: str = Field(..., alias="class")
    confidence: float
    track_id: int

class SpatialContext(BaseModel):
    atm_present: bool = False
    near_atm: bool = False
    
class IncidentData(BaseModel):
    frame_id: int
    detections: List[Detection]
    action_classification: ActionClassification
    spatial_context: SpatialContext

# --- Main Request Model ---
class IncidentCreate(BaseModel):
    camera_id: str = "cam_01"
    threat_level: str
    label: str
    confidence: float
    data: IncidentData # The detailed payload

class IncidentResponse(IncidentCreate):
    id: str
    created_at: datetime
    
    class Config:
        from_attributes = True
