import time
from src.config import (
    LOG_COOLDOWN, 
    GUN_CLASS_IDS, 
    KNIFE_CLASS_ID,
    ATM_CLASS_ID,
    BACKPACK_CLASS_ID,
    HANDBAG_CLASS_ID,
    PERSON_CLASS_ID
)
from src.utils.logger import setup_logger

class ThreatManager:
    def __init__(self):
        self.logger = setup_logger()
        self.last_log_time = {}  # { distinct_threat_key: timestamp }

    def determine_threat(self, action_label, all_class_ids):
        """
        Determines threat level based on your Research/Demo Matrix.
        Args:
            action_label (str): LSTM output ('shooting', 'violence', 'normal')
            all_class_ids (list): List of all YOLO class IDs detected in the frame.
        """
        # --- 1. PRE-PROCESS CONTEXT ---
        threat_level = "SAFE"
        box_color = (0, 255, 0)  # Green (Safe)

        # Context Flags (True/False)
        has_gun = any(c in GUN_CLASS_IDS for c in all_class_ids)
        has_knife = KNIFE_CLASS_ID in all_class_ids
        has_atm = ATM_CLASS_ID in all_class_ids
        has_bag = (BACKPACK_CLASS_ID in all_class_ids) or (HANDBAG_CLASS_ID in all_class_ids)
        
        person_count = all_class_ids.count(PERSON_CLASS_ID)
        is_aggressive = action_label in ["shooting", "violence"]

        # --- 2. THREAT MATRIX LOGIC ---

        # PRIORITY 1: Active Weapon Attack (Visual + Behavior)
        if (has_gun or has_knife) and is_aggressive:
            threat_level = "CRITICAL: ACTIVE ATTACK"
            box_color = (0, 0, 255)  # Red

        # PRIORITY 2: Armed Robbery (Weapon + ATM Context)
        elif (has_gun or has_knife) and has_atm:
            threat_level = "CRITICAL: ARMED ROBBERY (ATM)"
            box_color = (0, 0, 255)  # Red

        # PRIORITY 3: Brandishing Firearm (Gun + Passive)
        elif has_gun:
            threat_level = "CRITICAL: GUN DETECTED"
            box_color = (0, 0, 255)  # Red

        # PRIORITY 4: Brandishing Blade (Knife + Passive)
        elif has_knife:
            threat_level = "HIGH: KNIFE DETECTED"
            box_color = (0, 165, 255)  # Orange

        # PRIORITY 5: Bag Snatching (Bag + Violence)
        elif has_bag and is_aggressive:
            threat_level = "WARNING: POTENTIAL THEFT"
            box_color = (0, 255, 255)  # Yellow

        # PRIORITY 6: Physical Fight (No Weapon + Aggressive Motion)
        elif is_aggressive:
            threat_level = "WARNING: PHYSICAL FIGHT"
            box_color = (0, 255, 255)  # Yellow

        # PRIORITY 7: Shoulder Surfing (2+ Persons + ATM)
        elif has_atm and person_count >= 2:
            threat_level = "WARNING: SUSPICIOUS ACTIVITY (ATM)"
            box_color = (0, 255, 255)  # Yellow

        # PRIORITY 8: Suspicious Stance (No Weapon + Shooting Stance)
        elif action_label == "shooting":
            threat_level = "WARNING: SUSPICIOUS STANCE"
            box_color = (0, 255, 255)  # Yellow

        return threat_level, box_color

    def log_threat(self, track_id, source_name, action_label, threat_level):
        """
        Logs the threat with rate limiting.
        """
        if "SAFE" in threat_level:
            return

        # Create unique key to limit logs per threat type per camera
        log_key = f"{source_name}_{threat_level}"
        
        current_time = time.time()
        last_time = self.last_log_time.get(log_key, 0)
        
        if (current_time - last_time) > LOG_COOLDOWN:
            log_msg = (
                f"THREAT LOG | Source: {source_name} | ID: {track_id} | "
                f"Action: {action_label} | Level: {threat_level}"
            )
            
            if "CRITICAL" in threat_level:
                self.logger.critical(log_msg)
            elif "HIGH" in threat_level:
                self.logger.error(log_msg)
            else:
                self.logger.warning(log_msg)
            
            self.last_log_time[log_key] = current_time