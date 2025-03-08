import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(simple_formatter)

    # Create file handlers
    # Main log file
    main_log_file = log_dir / 'backend.log'
    file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Workload specific log file
    workload_log_file = log_dir / 'workload.log'
    workload_handler = RotatingFileHandler(
        workload_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    workload_handler.setLevel(logging.DEBUG)
    workload_handler.setFormatter(detailed_formatter)

    # Configure workload logger specifically
    workload_logger = logging.getLogger('app.services.workload_service')
    workload_logger.addHandler(workload_handler)
    workload_logger.propagate = True  # Allow logs to propagate to root logger as well

    # Add handlers to root logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Set levels for specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Set specific loggers to DEBUG level
    logging.getLogger('app.services.training_service').setLevel(logging.DEBUG)
    logging.getLogger('app.services.metrics_service').setLevel(logging.INFO)
    logging.getLogger('app.services.k8s_service').setLevel(logging.INFO)
    logging.getLogger('app.services.diagnosis_service').setLevel(logging.DEBUG)
    logging.getLogger('app.services.workload_service').setLevel(logging.INFO)
    logging.getLogger('app.api').setLevel(logging.INFO)

    # Quiet some noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('kubernetes').setLevel(logging.WARNING)
    
    # Log startup information
    logger.info("Logging system initialized")
    logger.info(f"Log directory: {log_dir.absolute()}")
    logger.info(f"Main log file: {main_log_file}")
    logger.info(f"Workload log file: {workload_log_file}")
    
    # Log environment information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Environment variables: OB_HOST={os.getenv('OB_HOST')}, OB_PORT={os.getenv('OB_PORT')}")
    
    # Test debug logging
    logger.debug("Debug logging is enabled - if you see this message, DEBUG level logging is working") 