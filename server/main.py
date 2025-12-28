import logging
import json
import os
import shutil
import uuid
from datetime import datetime
from fastapi import FastAPI, Depends, BackgroundTasks, File, UploadFile, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from . import models, schemas, database
from .services import telegram_service, email_service


# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartVisionServer")

# Initialize DB
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="SmartVision Backend", version="1.0.0")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()}")
    # logger.error(f"Body: {exc.body}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


def process_notifications(incident: schemas.IncidentCreate, image_path: str = None):
    msg = f" **SMARTVISION SECURITY ALERT** \n"
    msg += f"\n**Threat**: {incident.threat_level}\n"

    weapon_found = False
    if (
        "WEAPON" in incident.threat_level
        or "SHOOTER" in incident.threat_level
        or "KNIFE" in incident.threat_level
    ):
        detections = incident.data.detections or []
        for d in detections:
            if d.class_.lower() in ["gun", "knife"]:
                msg += f"**Weapon Detected**: {d.class_.upper()} ({d.confidence:.2f})\n"
                weapon_found = True

    if not weapon_found and (
        "CRITICAL" in incident.threat_level or "HIGH" in incident.threat_level
    ):
        if incident.label != "normal":
            msg += f"**Detected Action**: {incident.label.upper()} ({incident.confidence:.2f})\n"

    if "FIGHTING" in incident.threat_level or "VIOLENCE" in incident.threat_level:
        msg += f"**Activity**: VIOLENCE/FIGHTING ({incident.confidence:.2f})\n"

    if "ATM" in incident.threat_level:
        msg += f"**Context**: ATM Activity\n"

    msg += f"**Camera**: {incident.camera_id}\n"
    msg += f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    telegram_service.send_telegram_alert(msg, image_path)

    if "CRITICAL" in incident.threat_level or "HIGH" in incident.threat_level:
        email_service.send_email_alert(
            subject=f" Threat Alert: {incident.threat_level}",
            body=msg.replace("*", ""),  # Remove markdown for plain text email
            image_path=image_path
        )


@app.post("/api/v1/incidents", response_model=schemas.IncidentResponse)
def create_incident(
    incident_data: str = Form(...),
    file: UploadFile = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(database.get_db),
):
    try:
        data = json.loads(incident_data)
        incident = schemas.IncidentCreate(**data)
    except Exception as e:
        logger.error(f"Failed to parse incident_data: {e}")
        raise RequestValidationError([{"loc": ("body", "incident_data"), "msg": "Invalid JSON"}])
        
    logger.info(f"Received incident: {incident.label} ({incident.threat_level})")

    file_path = None
    if file:
        try:
            # Create unique filename
            ext = file.filename.split('.')[-1] if '.' in file.filename else "jpg"
            filename = f"{uuid.uuid4()}.{ext}"
            os.makedirs("server/static/alerts", exist_ok=True) # Ensure directory exists
            file_path = os.path.join("server/static/alerts", filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Saved snapshot to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")

    db_incident = models.Incident(
        camera_id=incident.camera_id,
        threat_level=incident.threat_level,
        label=incident.label,
        confidence=incident.confidence,
        data=incident.data.dict(by_alias=True),
        image_path=file_path
    )
    db.add(db_incident)
    db.commit()
    db.refresh(db_incident)

    if incident.threat_level != "SAFE":
        background_tasks.add_task(process_notifications, incident, file_path)

    return db_incident


@app.get("/api/v1/incidents", response_model=list[schemas.IncidentResponse])
def get_incidents(
    skip: int = 0, limit: int = 50, db: Session = Depends(database.get_db)
):
    incidents = (
        db.query(models.Incident)
        .order_by(models.Incident.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return incidents


@app.get("/")
def health_check():
    return {"status": "running"}
