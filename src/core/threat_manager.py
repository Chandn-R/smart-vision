import time
from src.config import (
    LOGGABLE_THREATS, LOG_COOLDOWN, GUN_CLASS_IDS, KNIFE_CLASS_ID
)

from src.utils.logger import setup_logger

class ThreatManager:
    def __init__(self):
        self.logger = setup_logger()
        self.last_log_time = {}  # { action_label: timestamp }

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
                threat_level = "WARN: SUSPICIOUS STANCE"
            else:
                threat_level = "WARN: FIGHTING"
            box_color = (0, 255, 255)  # Yellow
        
        # 4. WARNING (Loitering at ATM)
        elif has_atm and duration > ATM_LOITERING_THRESHOLD:
            threat_level = "WARN: LOITERING AT ATM"
            box_color = (0, 255, 255)  # Yellow
        
        # 5. WARNING (Unattended Baggage - Global)
        elif is_unattended_bag:
            threat_level = "WARN: UNATTENDED BAGGAGE"
            box_color = (0, 255, 255) # Yellow
            
        return threat_level, box_color

    def log_threat(self, track_id, source_name, action_label, action_prob, threat_level, frame_id=0, detections=[], spatial_context={}):
        """
        Logs the threat if conditions are met (rate limiting, etc.).
        """
        if threat_level == "SAFE":
            return

        # Always log to console/file if it matches basic criteria
        if action_label in LOGGABLE_THREATS or "CRITICAL" in threat_level or "HIGH" in threat_level or "WARN" in threat_level:
            current_time = time.time()
            # Rate limiting by threat type
            last_time = self.last_log_time.get(action_label, 0)
            
            if (current_time - last_time) > LOG_COOLDOWN:
                log_msg = f"THREAT DETECTED | Source: {source_name} | ID: {track_id} | Action: {action_label.upper()} ({action_prob:.2f}) | Level: {threat_level}"
                
                if "CRITICAL" in threat_level:
                    self.logger.critical(log_msg)
                elif "HIGH" in threat_level:
                    self.logger.error(log_msg)
                else:
                    self.logger.warning(log_msg)
                
                self.last_log_time[action_label] = current_time

                # --- SEND TO SERVER ---
                from src.config import SERVER_URL
                if SERVER_URL:
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
                    self._send_alert_async(SERVER_URL, payload)

    def _send_alert_async(self, url, payload):
        import threading
        import requests
        import json
        
        def _send():
            try:
                headers = {'Content-Type': 'application/json'}
                requests.post(url, data=json.dumps(payload), headers=headers, timeout=2)
            except Exception as e:
                # Silently fail or log debug to avoid spamming console
                pass 

        threading.Thread(target=_send, daemon=True).start()
