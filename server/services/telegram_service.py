import requests
import os
from ..config import settings
import logging

logger = logging.getLogger(__name__)

def send_telegram_alert(message: str, image_path: str = None):
    token = settings.TELEGRAM_BOT_TOKEN
    chat_id = settings.TELEGRAM_CHAT_ID
    
    if not token or not chat_id:
        logger.warning("Telegram token or chat_id not set. Skipping notification.")
        return

    try:
        url = f"https://api.telegram.org/bot{token}/"
        
        if image_path and os.path.exists(image_path):
            url += "sendPhoto"
            # Telegram 'caption' is the message for photos
            # Note: Caption length limit is 1024 chars, ensure message fits or truncate
            payload = {"chat_id": chat_id, "caption": message[:1000], "parse_mode": "Markdown"}
            with open(image_path, "rb") as photo:
                files = {"photo": photo}
                response = requests.post(url, data=payload, files=files, timeout=10)
        else:
            url += "sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
            response = requests.post(url, json=payload, timeout=5)

        response.raise_for_status()
        logger.info("Telegram notification sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
