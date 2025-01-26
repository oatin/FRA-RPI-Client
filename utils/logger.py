import logging
from typing import Optional

def get_logger(name: str, file_path: Optional[str] = None, level: Optional[int] = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if file_path:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger