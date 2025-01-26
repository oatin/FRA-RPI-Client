import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    BASE_URL: str = os.getenv('BASE_URL')
    MODEL_DOWNLOAD_URL: str = os.getenv('MODEL_DOWNLOAD_URL')
    MAP_DOWNLOAD_URL: str = os.getenv('MAP_DOWNLOAD_URL')
    API_USERNAME: str = os.getenv('API_USERNAME')
    API_PASSWORD: str = os.getenv('API_PASSWORD')
    
    # JWT Configuration
    TOKEN_ENDPOINT: str = '/api/token/'
    TOKEN_EXPIRY_MINUTES: int = 55  # Set to 5 minutes less than actual expiry
    
    # Path Configuration
    MODELS_DIR: str = os.getenv('MODELS_DIR')
    
    # Service Configuration
    CHECK_INTERVAL: int = int(os.getenv('CHECK_INTERVAL'))  # seconds
    LOG_LEVEL: str = os.getenv('LOG_LEVEL')
    
    # Request Configuration
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT'))  # seconds
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES'))
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_')
        }