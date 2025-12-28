import logging
from datetime import datetime
from fastapi import FastAPI, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from . import models, schemas, database
from .services import telegram_service, email_service


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartVisionServer")

# Create DB Tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="SmartVision Backend", version="1.0.0")


def process_notifications(incident: schemas.IncidentCreate):
    msg = f" **SMARTVISION ALERT** \n"
    msg += f"**Threat**: {incident.threat_level}\n"

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
            msg += f"**Action**: {incident.label.upper()} ({incident.confidence:.2f})\n"

    if "FIGHTING" in incident.threat_level or "VIOLENCE" in incident.threat_level:
        msg += f"**Activity**: VIOLENCE/FIGHTING ({incident.confidence:.2f})\n"

    if "ATM" in incident.threat_level:
        msg += f"**Context**: ATM Activity\n"

    msg += f"**Camera**: {incident.camera_id}\n"
    msg += f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    telegram_service.send_telegram_alert(msg)

    if "CRITICAL" in incident.threat_level or "HIGH" in incident.threat_level:
        email_service.send_email_alert(
            subject=f" Threat Alert: {incident.threat_level}",
            body=msg.replace("*", ""),  # Remove markdown for plain text email
        )


@app.post("/api/v1/incidents", response_model=schemas.IncidentResponse)
def create_incident(
    incident: schemas.IncidentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db),
):
    logger.info(f"Received incident: {incident.label} ({incident.threat_level})")

    db_incident = models.Incident(
        camera_id=incident.camera_id,
        threat_level=incident.threat_level,
        label=incident.label,
        confidence=incident.confidence,
        data=incident.data.dict(by_alias=True),
    )
    db.add(db_incident)
    db.commit()
    db.refresh(db_incident)

    if incident.threat_level != "SAFE":
        background_tasks.add_task(process_notifications, incident)

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
