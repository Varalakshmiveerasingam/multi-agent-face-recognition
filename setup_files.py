import os

detection = """import cv2
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
"""

main = """import cv2
import json
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from agents.detection_agent import FaceDetectionAgent
from agents.recognition_agent import FaceRecognitionAgent
from agents.emotion_agent import EmotionAnalysisAgent
from agents.liveness_agent import LivenessDetectionAgent
from agents.orchestrator import OrchestratorAgent
from agents.tracker_agent import FaceTrackerAgent

detection = FaceDetectionAgent()
recognition = FaceRecognitionAgent()
emotion = EmotionAnalysisAgent()
liveness = LivenessDetectionAgent()
orchestrator = OrchestratorAgent(recognition, emotion, liveness)
tracker = FaceTrackerAgent()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[*] Warming up camera...")
for _ in range(15):
    cap.read()
    time.sleep(0.05)

print("[*] MULTI-AGENT FACE RECOGNITION STARTED - Press Q to quit")
alert_display = []
fail_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        fail_count += 1
        if fail_count > 30:
            print("[ERROR] Camera lost.")
            break
        time.sleep(0.05)
        continue
    fail_count = 0

    try:
        output = detection.run(frame)
        faces = output['faces']
        face_count = len(faces)
        results = []
        identities = []
        for face_data in faces:
            result = orchestrator.orchestrate(face_data)
            results.append(result)
            identities.append(result['recognition']['identity'])

        track_map = tracker.update(faces, identities)

        for i, (face_data, result) in enumerate(zip(faces, results)):
            x, y, w, h = face_data['box']
            decision = result['decision']
            identity = result['recognition']['identity']
            emotion_label = result['emotion'].get('dominant_emotion', 'unknown')
            score = result['recognition']['score']
            source = face_data.get('source', '?')
            track_id = None
            for tid, idx2 in track_map.items():
                if idx2 == i:
                    track_id = tid
                    break
            if 'ACCEPT' in decision:
                color = (0, 255, 0)
                bg_color = (0, 150, 0)
            elif 'ESCALATE' in decision:
                color = (0, 165, 255)
                bg_color = (0, 100, 160)
            else:
                color = (0, 0, 255)
                bg_color = (150, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            top_y = max(42, y)
            cv2.rectangle(frame, (x, top_y-42), (x+w, top_y), bg_color, -1)
            cv2.putText(frame, f"ID#{track_id} {identity}", (x+4, top_y-24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv2.putText(frame, f"{decision} {score:.0%}", (x+4, top_y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            bot_y = min(frame.shape[0]-22, y+h)
            cv2.rectangle(frame, (x, bot_y), (x+w, bot_y+22), bg_color, -1)
            cv2.putText(frame, f"Emotion:{emotion_label} | {source}", (x+4, bot_y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
            if identity == 'UNKNOWN_PERSON' and track_id and tracker.get_alert_status(track_id, identity):
                alert_display.append({'msg': f"ALERT! Unknown person ID#{track_id} entered!", 'time': time.time()})

        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (430, 65), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "MULTI-AGENT FACE RECOGNITION", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"Faces: {face_count}  |  Tracked IDs: {len(tracker.tracked_faces)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        now = time.time()
        alert_display = [a for a in alert_display if now - a['time'] < 4.0]
        for j, alert in enumerate(alert_display[-3:]):
            ay = 100 + j*36
            cv2.rectangle(frame, (0, ay-24), (560, ay+8), (0,0,180), -1)
            cv2.putText(frame, alert['msg'], (10, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    except Exception as e:
        print(f"[WARN] Frame error: {e}")

    cv2.imshow('Multi-Agent Face Recognition - ALL FACES', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
with open('audit_log.json', 'w') as f:
    json.dump(orchestrator.audit_log, f, indent=2)
with open('tracker_log.json', 'w') as f:
    json.dump(tracker.alert_log, f, indent=2)
print("[*] Done! Logs saved.")
"""

files = {
    'agents/detection_agent.py': detection,
    'main.py': main,
}

for path, content in files.items():
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'[+] Created: {path}')

print('\n[DONE] All files ready! Now run: python main.py')