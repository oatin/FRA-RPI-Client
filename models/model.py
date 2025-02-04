import tensorflow as tf
from utils.logger import get_logger
from typing import Optional, Any
import numpy as np
from PIL import Image

import cv2
import dlib
from imutils import face_utils

logger = get_logger(__name__,file_path="logs/model.log")

class face_recognition_model:
    def __init__(self):
        self.model = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        
    def load_model(self, model_path: str) -> Optional[tf.keras.Model]:
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def preprocess(self, frame):
        # Step 1: Face detection (keeping your existing code)
        gray_dlib = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_dlib = self.detector(gray_dlib, 0)
        
        face_images = []  # จะเก็บใบหน้าที่ crop แล้ว
        
        for face in faces_dlib:
            # ทำ face landmark detection
            shape = self.predictor(gray_dlib, face)
            shape = face_utils.shape_to_np(shape)
            
            # วาด landmarks และกรอบหน้า (สำหรับ visualization)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Step 2: Crop face and preprocess for MobileNetV2
            face_img = frame[y1:y2, x1:x2]
            
            # Resize to MobileNetV2 input size (224x224)
            face_img = cv2.resize(face_img, (224, 224))
            
            # Convert to RGB (MobileNetV2 expects RGB)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Scale pixel values to [0, 1]
            face_img = face_img.astype('float32') / 255.0
            
            # Normalize using ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            face_img = (face_img - mean) / std
            
            # Add batch dimension
            face_img = np.expand_dims(face_img, axis=0)
            
            face_images.append(face_img)
        
        return frame, face_images

    def predict(self, input_data: Any) -> Any:
        if self.model is None:
            logger.error("Model not loaded")
            return None
            
        # Get preprocessed frame and face images
        frame, face_images = self.preprocess(input_data)
        
        predictions = []
        for face_img in face_images:
            prediction = self.model.predict(face_img, verbose=0)
            predictions.append(prediction)
        
        return frame, predictions