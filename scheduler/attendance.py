import requests
import json
from datetime import datetime
from typing import List, Dict, Any
from time import time
import os

from utils.logger import get_logger

logger = get_logger(__name__, file_path="logs/attendance.log")


class OfflineHandler:
    def __init__(self, offline_storage_file: str = "offline_data.json"):
        offline_dir = os.path.dirname(offline_storage_file)
        if offline_dir:
            os.makedirs(offline_dir, exist_ok=True)
            
        self.offline_storage_file = offline_storage_file
        self.offline_data = self._load_offline_data()

    def _load_offline_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.offline_storage_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_offline_data(self):
        with open(self.offline_storage_file, "w", encoding="utf-8") as file:
            json.dump(self.offline_data, file, indent=4, ensure_ascii=False) 

    def is_online(self) -> bool:
        test_urls = ["http://www.google.com", "http://1.1.1.1", "http://api.openai.com"]
        for url in test_urls:
            try:
                requests.get(url, timeout=5)
                return True
            except requests.ConnectionError:
                continue
        return False  

    def save_offline(self, data: Dict[str, Any]):
        self.offline_data.append(data)
        self._save_offline_data()
        logger.info("Data saved offline.")

    def sync_offline_data(self, post_attendance_func: callable):
        if self.is_online():
            successful_syncs = [data for data in self.offline_data if post_attendance_func(data) is not None]
            
            if successful_syncs:
                logger.info(f"Synced {len(successful_syncs)} records.")
            
            self.offline_data = [data for data in self.offline_data if data not in successful_syncs]
            self._save_offline_data()
            logger.info("Offline data synced.")
        else:
            logger.warning("System is offline. Cannot sync data.")


class AttendanceProcessor:
    def __init__(self, offline_handler, post_attendance_func):
        self.offline_handler = offline_handler
        self.post_attendance = post_attendance_func
        self.sent_records_file = f"logs/attendance/records/sent_records_data_{datetime.now().date().isoformat()}.json"
        
        os.makedirs(os.path.dirname(self.sent_records_file), exist_ok=True)
        
        self.sent_records = self._load_sent_records()
        self.last_sent_time = 0
        self.throttle_interval = 5  

    def _load_sent_records(self):
        try:
            with open(self.sent_records_file, "r") as file:
                return set(tuple(record) for record in json.load(file))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    def _save_sent_records(self):
        with open(self.sent_records_file, "w") as file:
            json.dump(list(self.sent_records), file)

    def postprocess(self, prediction: int, course_id: int, schedule_id: int, device_id: int) -> int:
        current_time = time()

        if current_time - self.last_sent_time < self.throttle_interval:
            logger.info("Throttling: Waiting to send data again.")
            return prediction

        record_key = (prediction, course_id, schedule_id)

        if record_key in self.sent_records:
            logger.info("Data already sent. Skipping.")
            return prediction

        attendance_data = {
            "schedule": schedule_id,
            "student": prediction,
            "course": course_id,
            "date": datetime.now().date().isoformat(),
            "time": datetime.now().time().isoformat(),
            "status": "present",
            "device": device_id,
        }

        if self.offline_handler.is_online():
            if self.post_attendance(attendance_data) is not None:
                logger.info("Data sent to API successfully.")
                self.sent_records.add(record_key)
                self.last_sent_time = current_time
                self._save_sent_records()
            else:
                logger.warning("Failed to send data to API. Saving offline.")
                self.offline_handler.save_offline(attendance_data)  
        else:
            logger.warning("System is offline. Saving data locally.")
            self.offline_handler.save_offline(attendance_data) 

        return prediction

    def reset_sent_records(self):
        self.sent_records.clear()
        self._save_sent_records()
        logger.info("Reset sent_records for new course.")