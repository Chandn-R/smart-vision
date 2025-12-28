import time
import cv2
import logging
from datetime import datetime
from src.config import (
    LOGGABLE_THREATS, LOG_COOLDOWN, GUN_CLASS_IDS, KNIFE_CLASS_ID
)

from src.utils.logger import setup_logger

class ThreatManager:
    def __init__(self):
        self.logger = setup_logger()
        self.last_log_time = {}  # { action_label: timestamp }
        self.threat_logs = {}    # { track_id: (timestamp, distinct_threat_level) }
        self.log_cooldown = LOG_COOLDOWN
        self.csv_writer = None
        self.log_file = None

    def determine_threat(self, action_label, visible_objects, duration=0.0, is_unattended_bag=False):
        """
        Determines the threat level based on the following matrix:
        - Critical: (Shooting/Violence) + (Gun/Knife) OR (Shooting/Violence) + ATM
        - High:     (Any Action) + (Gun/Knife)
        - Warning:  (Shooting/Violence) + No Weapon
        - Warning:  (Normal) + ATM + Duration > LOITERING_THRESHOLD (Loitering)
        - Warning:  Unattended Baggage
        - Safe:     Otherwise
        """
        threat_level = "SAFE"
        box_color = (0, 255, 0)  # Green

        from src.config import ATM_CLASS_ID, GUN_CLASS_IDS, KNIFE_CLASS_ID, ATM_LOITERING_THRESHOLD

        has_gun = any(w in GUN_CLASS_IDS for w in visible_objects)
        has_knife = KNIFE_CLASS_ID in visible_objects
        has_atm = ATM_CLASS_ID in visible_objects
        has_weapon = has_gun or has_knife

        is_violent_action = action_label in ["shooting", "violence"]

        # 1. CRITICAL
        if is_violent_action and (has_weapon or has_atm):
            if has_gun:
                threat_level = "CRITICAL: SHOOTER"
            elif has_knife:
                threat_level = "CRITICAL: KNIFE ATTACK"
            elif has_atm:
                 threat_level = "CRITICAL: ATM ROBBERY"
            box_color = (0, 0, 255)  # Red

        # 2. HIGH (Weapon Visible)
        elif has_weapon:
            threat_level = "HIGH: WEAPON DETECTED"
            box_color = (0, 165, 255)  # Orange

        # 3. WARNING (Violence w/o Weapon)
        elif is_violent_action:
            if action_label == "shooting":
                threat_level = "WARNING: SUSPICIOUS STANCE"
            else:
                threat_level = "WARNING: FIGHTING"
            box_color = (0, 255, 255)  # Yellow
        
        # 4. WARNING (Loitering at ATM)
        elif has_atm and duration > ATM_LOITERING_THRESHOLD:
            threat_level = "WARNING: LOITERING AT ATM"
            box_color = (0, 255, 255)  # Yellow
        
        # 5. WARNING (Unattended Baggage - Global)
        elif is_unattended_bag:
            threat_level = "WARNING: UNATTENDED BAGGAGE"
            box_color = (0, 255, 255) # Yellow
            
        return threat_level, box_color

    def log_threat(self, track_id, source_name, action_label, action_prob, threat_level, frame_id=0, detections=[], spatial_context={}, frame=None):
        """
        Logs threat to console/file and conditionally sends to server.
        """
        current_time = time.time()
        
        if threat_level == "SAFE":
            return

        # Global Debounce logic (prevent spamming same threat TYPE regardless of ID)
        # Check if we already logged this specific threat level recently
        last_time = self.threat_logs.get(threat_level, 0)
        if current_time - last_time < self.log_cooldown:
            return

        # Log to Console/File
        if "CRITICAL" in threat_level:
            logging.critical(f"THREAT DETECTED | Source: {source_name} | ID: {track_id} | Action: {action_label} ({action_prob:.2f}) | Level: {threat_level}")
        elif "WARNING" in threat_level:
             logging.warning(f"THREAT DETECTED | Source: {source_name} | ID: {track_id} | Action: {action_label} ({action_prob:.2f}) | Level: {threat_level}")
        else:
             logging.error(f"THREAT DETECTED | Source: {source_name} | ID: {track_id} | Action: {action_label} ({action_prob:.2f}) | Level: {threat_level}")

        # Update global cooldown for this threat level
        self.threat_logs[threat_level] = current_time
        if self.csv_writer: # Only write if csv_writer was successfully initialized
            self.csv_writer.writerow([datetime.now(), source_name, track_id, action_label, f"{action_prob:.2f}", threat_level])
            self.log_file.flush()

        # --- SEND TO SERVER ---
        from src.config import SERVER_URL
        if SERVER_URL and threat_level != "SAFE":
            # Construct Payload
            payload = {
                "camera_id": source_name,
                "threat_level": threat_level,
                "label": action_label.upper(),
                "confidence": float(action_prob),
                "data": {
                    "frame_id": frame_id,
                    "detections": detections,
                    "action_classification": {
                        "class": action_label,
                        "confidence": float(action_prob),
                        "track_id": track_id
                    },
                    "spatial_context": spatial_context
                }
            }
            
            # Prepare Image if Critical/High
            image_data = None
            if frame is not None and ("CRITICAL" in threat_level or "HIGH" in threat_level):
                try:
                     # Encode frame to jpg
                     success, buffer = cv2.imencode(".jpg", frame)
                     if success:
                         image_data = buffer.tobytes()
                except Exception as e:
                    logging.error(f"Failed to encode frame: {e}")

            self._send_alert_async(SERVER_URL, payload, image_data)

    def _send_alert_async(self, url, payload, image_data=None):
        import threading
        import requests
        import json
        
        def _send():
            try:
                # Always use multipart/form-data because server Endpoint has File(...)
                # Requests only sends multipart if 'files' is provided.
                files = {}
                if image_data:
                    files["file"] = ("alert.jpg", image_data, "image/jpeg")
                else:
                    # Force multipart by adding a dummy file that server will ignore
                    # or explicitly empty 'file' field? 
                    # If I send ('file', (None, '')) requests might send it.
                    # Safest: Send a dummy field that isn't 'file' to force headers
                    files["_dummy"] = ("dummy", b"", "text/plain")

                data = {
                    "incident_data": json.dumps(payload)
                }
                
                requests.post(url, data=data, files=files, timeout=5)
                    
            except Exception as e:
                # Silently fail or log debug
                logging.debug(f"Failed to send alert to server: {e}")

        threading.Thread(target=_send, daemon=True).start()