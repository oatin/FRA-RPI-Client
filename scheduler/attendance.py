import requests
import json
from datetime import datetime
from typing import List, Dict, Any
from time import time

from utils.logger import get_logger

logger = get_logger(__name__,file_path="logs/attendance.log")

class OfflineHandler:
    def __init__(self, offline_storage_file: str = "offline_data.json"):
        self.offline_storage_file = offline_storage_file
        self.offline_data = self._load_offline_data()

    def _load_offline_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.offline_storage_file, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_offline_data(self):
        with open(self.offline_storage_file, "w") as file:
            json.dump(self.offline_data, file)

    def is_online(self) -> bool:
        try:
            requests.get("http://www.google.com", timeout=5)
            return True
        except requests.ConnectionError:
            return False

    def save_offline(self, data: Dict[str, Any]):
        self.offline_data.append(data)
        self._save_offline_data()
        print("Data saved offline.")

    def sync_offline_data(self, post_attendance_func: callable):
        if self.is_online():
            successful_syncs = []
            for data in self.offline_data:
                if post_attendance_func(data) is not None:
                    successful_syncs.append(data)
                    print(f"Synced data: {data}")
                else:
                    print(f"Failed to sync data: {data}")

            self.offline_data = [data for data in self.offline_data if data not in successful_syncs]
            self._save_offline_data()
            print("Offline data synced.")
        else:
            print("System is offline. Cannot sync data.")


class AttendanceProcessor:
    def __init__(self, offline_handler: OfflineHandler, post_attendance_func: callable):
        self.offline_handler = offline_handler
        self.post_attendance = post_attendance_func
        self.sent_records = set()  
        self.last_sent_time = 0  
        self.throttle_interval = 5  

    def postprocess(self, prediction: int, course_id: int, schedule_id: int, device_id: int) -> int:
        current_time = time()

        if current_time - self.last_sent_time < self.throttle_interval:
            print("Throttling: Waiting to send data again.")
            return prediction

        record_key = (prediction, course_id, schedule_id)

        if record_key in self.sent_records:
            print("Data already sent. Skipping.")
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
                print("Data sent to API successfully.")
                self.sent_records.add(record_key)  
                self.last_sent_time = current_time  
            else:
                print("Failed to send data to API. Saving offline.")
                self.offline_handler.save_offline(attendance_data)
        else:
            print("System is offline. Saving data locally.")
            self.offline_handler.save_offline(attendance_data)

        return prediction