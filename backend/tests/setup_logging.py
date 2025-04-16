import sys
import logging
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import random
from logging.handlers import RotatingFileHandler

class VerboseFilter(logging.Filter):
    def filter(self, record):
        # Filter out debug messages containing response body
        if record.levelno == logging.DEBUG and "response body:" in record.getMessage():
            return False
        return True

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    main_log_file = log_dir / 'test_model_split.log'
    file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=10*1024*1024,
        backupCount=2
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set the root logger level
    logger.setLevel(logging.DEBUG)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add filter to console handler
    console_handler.addFilter(VerboseFilter())

    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set specific loggers to higher level to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('kubernetes').setLevel(logging.WARNING)
    
    return logger

