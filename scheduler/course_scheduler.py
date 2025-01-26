from datetime import datetime
from typing import Optional, List, Dict
import os
from api.client import APIClient
from models.model import face_recognition_model
from utils.logger import get_logger

import numpy as np
import cv2
import json

logger = get_logger(__name__,file_path="logs/scheduler.log")

class CourseScheduler:
    def __init__(self, api_client: APIClient, device_id: str):
        self.api_client = api_client
        self.device_id = device_id
        self.current_model = None
        self.model_manager = face_recognition_model()
        self.models_dir = "course_models"
        self.label_map = None
        
    def is_course_time(self, schedule_item: Dict) -> bool:
        current_time = datetime.now().time()
        start_time = datetime.strptime(schedule_item['start_time'], '%H:%M:%S').time()
        end_time = datetime.strptime(schedule_item['end_time'], '%H:%M:%S').time()

        return start_time <= current_time <= end_time

    def load_course_model(self, course_id: int) -> bool:
        model_path = os.path.join(self.models_dir, f"model_{course_id}.keras")
        label_map_path = os.path.join(self.models_dir, f"model_{course_id}.json")

        if not os.path.exists(model_path):
            logger.error(f"Model not found for course {course_id}")
            return False
            
        self.current_model = self.model_manager.load_model(model_path)
        
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
        else:
            logger.error(f"label_map not found for course {course_id}")
            return False
        
        return self.current_model is not None

    def get_all_schedule(self):
        return self.api_client.get_schedule(self.device_id)
    
    def check_and_update_model(self) -> None:
        schedule_data = self.api_client.get_schedule(self.device_id)
        if not schedule_data or 'results' not in schedule_data:
            logger.error("Failed to get schedule data or invalid format")
            return

        current_day = datetime.now().strftime("%A")
        
        # check each schedule in results
        for schedule in schedule_data['results']:
            if (schedule['day_of_week'] == current_day and 
                self.is_course_time(schedule)):
                course_id = schedule['course']

                if self.load_course_model(course_id):
                    self.run_face_recognition(schedule['end_time'])
                    logger.info(f"Loaded and running model for course {course_id}")
                break

    def run_face_recognition(self, end_time: str) -> None:
        if self.current_model is None:
            logger.warning("No model currently loaded")
            return

        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("ไม่สามารถเปิดกล้องได้")
                return

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("ไม่สามารถอ่านภาพจากกล้องได้")
                    break
                
                result, predictions = self.model_manager.predict(frame)

                if predictions is not None and len(predictions) > 0:
                    pred_index = np.argmax(predictions)

                    if self.label_map is not None and str(pred_index) in self.label_map:
                        predicted_label = self.label_map[str(pred_index)]
                    else:
                        predicted_label = "Unknown" 

                    cv2.putText(result, f'Prediction: {predicted_label}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(result, 'No face detected', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow('Live Camera Feed', result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = datetime.now().strftime('%H:%M:%S')
                if current_time >= end_time:
                    logger.info("Reached end time. Closing camera.")
                    break

            cap.release()
            cv2.destroyAllWindows()

            logger.info(f"Face recognition completed.")
        except Exception as e:
            logger.error(f"Error running face recognition: {str(e)}")
