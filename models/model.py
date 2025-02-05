import tensorflow as tf
from utils.logger import get_logger
from typing import Optional, Any
import numpy as np

import cv2
import dlib
import os

logger = get_logger(__name__,file_path="logs/model.log")

class FaceRecognitionModel:
    def __init__(self):
        self.model = None
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_path = 'models/shape_predictor_68_face_landmarks.dat'

        if os.path.exists(self.landmark_path):
            self.predictor = dlib.shape_predictor(self.landmark_path)
        else:
            logger.error(f"Landmark model file {self.landmark_path} not found.")

    def load_model(self, model_path: str) -> Optional[tf.keras.Model]:
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        face_images = []

        for face in faces:
            shape = self.predictor(gray, face) if hasattr(self, 'predictor') else None
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_img = frame[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            face_images.append(face_img)

        return frame, face_images if face_images else None

    def predict(self, frame: Any) -> Any:
        if self.model is None:
            logger.error("Model not loaded")
            return frame, None

        frame, face_images = self.preprocess(frame)
        predictions = [self.model.predict(face, verbose=0) for face in face_images] if face_images else None
        return frame, predictions