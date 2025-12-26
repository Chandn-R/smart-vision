import time
from src.config import (
    LOGGABLE_THREATS, LOG_COOLDOWN, GUN_CLASS_IDS
)
from src.utils.logger import setup_logger

class ThreatManager:
    def __init__(self):
        self.logger = setup_logger()
        self.last_log_time = {}  # { action_label: timestamp }

    def determine_threat(self, action_label, visible_weapons):
        """
        Determines the threat level and display color based on action and visible weapons.
        Returns: (threat_level_string, color_rgb)
        """
        threat_level = "SAFE"
        box_color = (0, 255, 0)  # Green

        has_gun = any(w in GUN_CLASS_IDS for w in visible_weapons)
        
        if action_label == "shooting" and has_gun:
            threat_level = "CRITICAL: SHOOTER"
            box_color = (0, 0, 255)  # Red
        elif action_label == "violence" and (3 in visible_weapons): # 3 is Knife
            threat_level = "CRITICAL: KNIFE ATTACK"
            box_color = (0, 0, 255)  # Red
        elif action_label == "violence":
            threat_level = "HIGH: FIGHTING"
            box_color = (0, 165, 255)  # Orange
        elif action_label == "shooting":
            threat_level = "WARN: SUSPICIOUS STANCE"
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
