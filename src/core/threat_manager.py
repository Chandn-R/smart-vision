import time
from src.config import (
    LOGGABLE_THREATS, LOG_COOLDOWN, GUN_CLASS_IDS, KNIFE_CLASS_ID
)

from src.utils.logger import setup_logger

class ThreatManager:
    def __init__(self):
        self.logger = setup_logger()
        self.last_log_time = {}  # { action_label: timestamp }

    def determine_threat(self, action_label, visible_weapons):
        """
        Determines the threat level based on the following matrix:
        - Critical: (Shooting/Violence) + (Gun/Knife)
        - High:     (Normal/Scanning)   + (Gun/Knife)
        - Warning:  (Shooting/Violence) + (No Weapon)
        - Safe:     Otherwise
        """
        threat_level = "SAFE"
        box_color = (0, 255, 0)  # Green

        has_gun = any(w in GUN_CLASS_IDS for w in visible_weapons)
        has_knife = KNIFE_CLASS_ID in visible_weapons
        has_weapon = has_gun or has_knife

        is_violent_action = action_label in ["shooting", "violence"]

        if is_violent_action and has_weapon:
            # CRITICAL
            if has_gun:
                threat_level = "CRITICAL: SHOOTER"
            else:
                threat_level = "CRITICAL: KNIFE ATTACK"
            box_color = (0, 0, 255)  # Red

        elif has_weapon:
            # HIGH (Weapon detected but no violent action yet)
            threat_level = "HIGH: WEAPON DETECTED"
            box_color = (0, 165, 255)  # Orange

        elif is_violent_action:
            # WARNING (Violent action but no weapon visible)
            if action_label == "shooting":
                threat_level = "WARN: SUSPICIOUS STANCE"
            else:
                threat_level = "WARN: FIGHTING"
            box_color = (0, 255, 255)  # Yellow
            
        return threat_level, box_color

    def log_threat(self, track_id, source_name, action_label, action_prob, threat_level):
        """
        Logs the threat if conditions are met (rate limiting, etc.).
        """
        if threat_level == "SAFE":
            return

        if action_label in LOGGABLE_THREATS:
            current_time = time.time()
            # User requested "one type of log every 30 sec", so we track by action_label (threat type)
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
