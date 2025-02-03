import requests
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from utils.logger import get_logger
from config import Config
import time

logger = get_logger(__name__,file_path="logs/api.log")

class APIClient:
    def __init__(self, base_url: str = Config.BASE_URL):
        self.base_url = base_url
        self.access_token = None
        self.token_expiry = None
        
    def _get_token(self) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}{Config.TOKEN_ENDPOINT}",
                data={
                    "username": Config.API_USERNAME,
                    "password": Config.API_PASSWORD
                },
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            token_data = response.json()
            
            self.access_token = token_data['access']
            self.token_expiry = datetime.now() + timedelta(minutes=Config.TOKEN_EXPIRY_MINUTES)
            
            logger.info("Successfully obtained new access token")
            return True
            
        except Exception as e:
            logger.error(f"Failed to get access token: {str(e)}")
            return False
            
    def _ensure_valid_token(self) -> bool:
        if not self.access_token or not self.token_expiry or datetime.now() >= self.token_expiry:
            return self._get_token()
        return True
            
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        try:
            if not self._ensure_valid_token():
                return None

            headers = kwargs.get('headers', {})
            headers['Authorization'] = f"Bearer {self.access_token}"
            kwargs['headers'] = headers

            if 'timeout' not in kwargs:
                kwargs['timeout'] = Config.REQUEST_TIMEOUT

            url = f"{self.base_url}{endpoint}"

            for attempt in range(Config.MAX_RETRIES):
                try:
                    response = requests.request(method, url, **kwargs)

                    if response.status_code == 401 and self._get_token():
                        kwargs['headers']['Authorization'] = f"Bearer {self.access_token}"
                        response = requests.request(method, url, **kwargs)

                    response.raise_for_status()
                    return response.json()

                except requests.exceptions.RequestException as e:
                    if attempt == Config.MAX_RETRIES - 1:
                        # Log the error response
                        if hasattr(e, 'response') and e.response is not None:
                            logger.error(f"Error response: {e.response.text}")
                        raise
                    logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed after {Config.MAX_RETRIES} attempts: {str(e)}")
            return None

    def get_schedule(self, device_id: str) -> Optional[Dict]:
        return self._make_request('GET', '/api/Schedule/', params={'device_id': device_id})

    def get_course(self, course_id: str) -> Optional[Dict]:
        return self._make_request('GET', f'/api/courses/{course_id}/')

    def post_attendance(self, attendance_data: Dict) -> Optional[Dict]:
        required_fields = ["schedule", "student", "course", "date", "time", "status", "device"]
        for field in required_fields:
            if field not in attendance_data:
                logger.error(f"Missing required field: {field}")
                return None

        return self._make_request('POST', '/api/attendance/', json=attendance_data)

    def download_model(self, course_id: str, save_path: str) -> bool:
        try:
            if not self._ensure_valid_token():
                return False
                
            response = requests.get(
                f"{Config.MODEL_DOWNLOAD_URL}/{course_id}",
                headers={'Authorization': f"Bearer {self.access_token}"},
                stream=True,
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            return False
    
    def map_model(self, course_id: str, save_path: str) -> bool:
        try:
            if not self._ensure_valid_token():
                return False
                
            response = requests.get(
                f"{Config.MAP_DOWNLOAD_URL}/{course_id}",
                headers={'Authorization': f"Bearer {self.access_token}"},
                stream=True,
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
            
        except Exception as e:
            logger.error(f"Failed to download label_map: {str(e)}")
            return False