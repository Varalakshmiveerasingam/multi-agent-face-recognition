import cv2
import numpy as np

class LivenessDetectionAgent:
    def __init__(self):
        self.threshold = 0.15

    def check_liveness(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        liveness_score = min(variance / 100.0, 1.0)
        is_live = liveness_score > self.threshold
        return {
            'agent': 'LivenessAgent',
            'is_live': bool(is_live),
            'liveness_score': round(liveness_score, 4),
            'status': 'LIVE' if is_live else 'SPOOF'
        }

    def run(self, face_data):
        return self.check_liveness(face_data['roi'])
