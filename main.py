import schedule
import time
from api.client import APIClient
from scheduler.course_scheduler import CourseScheduler
from utils.logger import get_logger
from config import Config
import argparse

logger = get_logger(__name__, file_path="logs/app.log")

class FaceRecognitionService:
    def __init__(self, device_id: str):
        self.api_client = APIClient(Config.BASE_URL)
        self.scheduler = CourseScheduler(self.api_client, device_id)
        
    def start(self) -> None:
        logger.info("Starting Face Recognition Service...")

        schedule_data = self.scheduler.get_all_schedule()

        if schedule_data is None:
            logger.error("Failed to fetch schedule data. Retrying later...")
        elif "results" in schedule_data:
            for entry in schedule_data["results"]:
                course_id = entry.get("course") 
                if course_id and self.scheduler.check_model_update(course_id):
                    model_success = self.api_client.download_model(str(course_id), f"course_models/model_{course_id}.keras")
                    label_success = self.api_client.map_model(str(course_id), f"course_models/label_map_{course_id}.json")

                    if model_success and label_success:
                        with open(f"course_models/model_{course_id}.version", "w") as f:
                            f.write(str(self.api_client.get_model_version(str(course_id))))
                    else:
                        logger.error(f"Failed to download model or label map for course {course_id}")

        schedule.every(Config.CHECK_INTERVAL).seconds.do(self.scheduler.check_and_update_model)

        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(Config.CHECK_INTERVAL)
                continue  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", required=True, help="Specify the device ID")
    args = parser.parse_args()

    service = FaceRecognitionService(device_id=args.device_id)
    service.start()