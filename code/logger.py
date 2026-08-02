import logging
import json
from pathlib import Path
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage()
        }
        if hasattr(record, "telemetry"):
            log_record.update(record.telemetry)
        return json.dumps(log_record)

def get_telemetry_logger():
    logger = logging.getLogger("TelemetryLogger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / "telemetry.log")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
        
    return logger

telemetry = get_telemetry_logger()
