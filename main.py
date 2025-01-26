import schedule
import time
from api.client import APIClient
from scheduler.course_scheduler import CourseScheduler
from utils.logger import get_logger
from config import Config

logger = get_logger(__name__,file_path="logs/app.log")

class FaceRecognitionService:
    def __init__(self, device_id: str):
        self.api_client = APIClient(Config.BASE_URL)
        self.scheduler = CourseScheduler(self.api_client, device_id)
        
    def start(self) -> None:
        logger.info("Starting Face Recognition Service...")
        
        schedule_data = self.scheduler.get_all_schedule()

        if "results" in schedule_data:
            for entry in schedule_data["results"]:
                course_id = entry.get("course") 
                if course_id:
                    self.api_client.download_model(str(course_id), f"course_models/model_{course_id}.keras")
                    self.api_client.map_model(str(course_id), f"course_models/model_{course_id}.json")

        schedule.every(Config.CHECK_INTERVAL).seconds.do(
            self.scheduler.check_and_update_model
        )
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(Config.CHECK_INTERVAL)

if __name__ == "__main__":
    service = FaceRecognitionService(device_id="1")
    service.start()