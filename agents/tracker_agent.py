import cv2
import numpy as np
import time

class FaceTrackerAgent:
    def __init__(self):
        self.tracked_faces = {}
        self.next_id = 1
        self.alert_log = []
        self.unknown_alert_cooldown = {}

    def iou(self, boxA, boxB):
        ax,ay,aw,ah = boxA
        bx,by,bw,bh = boxB
        ix = max(ax, bx); iy = max(ay, by)
        ex = min(ax+aw, bx+bw); ey = min(ay+ah, by+bh)
        if ex < ix or ey < iy:
            return 0.0
        inter = (ex-ix)*(ey-iy)
        union = aw*ah + bw*bh - inter
        return inter/union if union > 0 else 0.0

    def update(self, faces, identities):
        current_ids = {}
        for i, (face_data, identity) in enumerate(zip(faces, identities)):
            box = face_data['box']
            matched_id = None
            best_iou = 0.3
            for tid, tinfo in self.tracked_faces.items():
                score = self.iou(box, tinfo['box'])
                if score > best_iou:
                    best_iou = score
                    matched_id = tid
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
                alert_msg = f"NEW FACE #{matched_id} DETECTED!"
                self.alert_log.append({'id': matched_id, 'identity': identity, 'time': time.strftime('%H:%M:%S'),'msg': alert_msg})
                print(f"[TrackerAgent] ALERT: {alert_msg}")
            self.tracked_faces[matched_id] = {'box': box, 'identity': identity, 'last_seen': time.time()}
            current_ids[matched_id] = i

        stale = [tid for tid, tinfo in self.tracked_faces.items() if time.time() - tinfo['last_seen'] > 3.0]
        for tid in stale:
            del self.tracked_faces[tid]

        return current_ids

    def get_alert_status(self, track_id, identity):
        now = time.time()
        if identity == 'UNKNOWN_PERSON':
            last = self.unknown_alert_cooldown.get(track_id, 0)
            if now - last > 5.0:
                self.unknown_alert_cooldown[track_id] = now
                return True
        return False
