import cv2
import numpy as np
from mtcnn import MTCNN

class FaceDetectionAgent:
    def __init__(self):
        self.mtcnn = MTCNN()
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.min_confidence = 0.60

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []
        h_frame, w_frame = frame.shape[:2]
        if h_frame < 50 or w_frame < 50:
            return []
        faces = []
        boxes_seen = []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mtcnn_results = self.mtcnn.detect_faces(rgb)
            for det in mtcnn_results:
                if det['confidence'] < self.min_confidence:
                    continue
                x, y, w, h = det['box']
                x, y = max(0, x), max(0, y)
                w = min(w, w_frame - x)
                h = min(h, h_frame - y)
                if w < 30 or h < 30:
                    continue
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                face_roi = cv2.resize(face_roi, (224, 224))
                faces.append({'roi': face_roi, 'box': (x,y,w,h),
                              'confidence': round(det['confidence'],3),
                              'keypoints': det.get('keypoints',{}),
                              'source': 'MTCNN'})
                boxes_seen.append((x,y,w,h))
        except Exception as e:
            print(f"[MTCNN] Error: {e}")
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            haar_faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50,50))
            for (hx, hy, hw, hh) in haar_faces:
                duplicate = any(abs(hx-bx)<50 and abs(hy-by)<50 for (bx,by,bw,bh) in boxes_seen)
                if duplicate:
                    continue
                face_roi = frame[hy:hy+hh, hx:hx+hw]
                if face_roi.size == 0:
                    continue
                face_roi = cv2.resize(face_roi, (224, 224))
                faces.append({'roi': face_roi, 'box': (hx,hy,hw,hh),
                              'confidence': 0.75, 'keypoints': {}, 'source': 'Haar'})
                boxes_seen.append((hx,hy,hw,hh))
        except Exception as e:
            print(f"[Haar] Error: {e}")
        return faces

    def run(self, frame):
        faces = self.detect(frame)
        print(f"[DetectionAgent] Faces found: {len(faces)}")
        return {'agent': 'DetectionAgent', 'faces_found': len(faces), 'faces': faces}
