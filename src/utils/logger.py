import logging
import os
import sys
from src.config import LOG_DIR

def setup_logger(name="SmartVision", log_file="threat_alerts.log"):
    """
    Sets up a logger that writes to both console and a file.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers are already added to avoid duplicate logs
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File Handler
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger
