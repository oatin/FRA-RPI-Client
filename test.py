import cv2
import dlib
import socket
import numpy as np
import pickle
import struct
from imutils import face_utils

def create_client_socket():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.0.107', 9999))
    return client_socket

def receive_frame(client_socket):
    data = b""
    payload_size = struct.calcsize("L")
    
    while len(data) < payload_size:
        data += client_socket.recv(4096)
    
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]
    
    while len(data) < msg_size:
        data += client_socket.recv(4096)
    
    frame_data = data[:msg_size]
    frame = pickle.loads(frame_data)
    return frame

def main():
    # Initialize detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    
    cap = cv2.VideoCapture(0)

    # client_socket = create_client_socket()
    
    # FPS calculation
    fps = 0
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        # Receive frame from stream
        # frame = receive_frame(client_socket)
        
        # Create copies for each detector
        frame_opencv = frame.copy()
        frame_dlib = frame.copy()
        
        # OpenCV Detection
        gray_opencv = cv2.cvtColor(frame_opencv, cv2.COLOR_BGR2GRAY)
        faces_opencv = face_cascade.detectMultiScale(gray_opencv, 1.1, 4)
        
        for (x, y, w, h) in faces_opencv:
            cv2.rectangle(frame_opencv, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # dlib Detection
        gray_dlib = cv2.cvtColor(frame_dlib, cv2.COLOR_BGR2GRAY)
        faces_dlib = detector(gray_dlib, 0)
        
        for face in faces_dlib:
            shape = predictor(gray_dlib, face)
            shape = face_utils.shape_to_np(shape)
            
            # Draw facial landmarks
            for (x, y) in shape:
                cv2.circle(frame_dlib, (x, y), 2, (0, 255, 0), -1)
            
            # Draw face rectangle
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame_dlib, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Combine frames side by side
        combined_frame = np.hstack((frame_opencv, frame_dlib))
        
        # Add labels
        cv2.putText(combined_frame, 'OpenCV', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, 'dlib', (frame.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Calculate FPS
        cur_time = cv2.getTickCount()
        time_diff = (cur_time - prev_time) / cv2.getTickFrequency()
        prev_time = cur_time
        fps = 1 / time_diff
        
        cv2.putText(combined_frame, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow('Face Detection Comparison', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # client_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
