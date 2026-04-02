import cv2
import json
import os
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from agents.detection_agent import FaceDetectionAgent
from agents.recognition_agent import FaceRecognitionAgent
from agents.emotion_agent import EmotionAnalysisAgent
from agents.liveness_agent import LivenessDetectionAgent
from agents.orchestrator import OrchestratorAgent
from agents.tracker_agent import FaceTrackerAgent

detection   = FaceDetectionAgent()
recognition = FaceRecognitionAgent()
emotion     = EmotionAnalysisAgent()
liveness    = LivenessDetectionAgent()
orchestrator = OrchestratorAgent(recognition, emotion, liveness)
tracker     = FaceTrackerAgent()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

print("[*] Warming up camera...")
time.sleep(2)
for _ in range(30):
    cap.read()
    time.sleep(0.1)
print("[*] SYSTEM ONLINE")

alert_display  = []
fail_count     = 0
PROCESS_EVERY  = 3
frame_count    = 0
last_results   = []
last_faces     = []
last_track_map = {}
smooth_boxes   = {}
smooth_labels  = {}
SMOOTH_WIN     = 6
scan_line_y    = 0
start_time     = time.time()
total_detected = 0
total_accepted = 0
total_rejected = 0

emoji_map = {
    'happy':    'HAPPY :)',
    'sad':      'SAD :(',
    'angry':    'ANGRY >:(',
    'surprise': 'SURPRISED :O',
    'fear':     'FEAR :|',
    'disgust':  'DISGUST',
    'neutral':  'NEUTRAL :|'
}

emotion_colors = {
    'happy':    (0, 255, 255),
    'sad':      (255, 100, 50),
    'angry':    (0, 0, 255),
    'surprise': (0, 200, 255),
    'fear':     (128, 0, 128),
    'disgust':  (0, 128, 0),
    'neutral':  (200, 200, 200)
}

def smooth_box(track_id, new_box):
    if track_id not in smooth_boxes:
        smooth_boxes[track_id] = []
    smooth_boxes[track_id].append(new_box)
    if len(smooth_boxes[track_id]) > SMOOTH_WIN:
        smooth_boxes[track_id].pop(0)
    boxes = smooth_boxes[track_id]
    return (
        int(sum(b[0] for b in boxes)/len(boxes)),
        int(sum(b[1] for b in boxes)/len(boxes)),
        int(sum(b[2] for b in boxes)/len(boxes)),
        int(sum(b[3] for b in boxes)/len(boxes))
    )

def draw_corner_box(img, x, y, w, h, color, thickness=2, corner_len=20):
    # Draw fancy corner brackets instead of full rectangle
    pts = [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]
    dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
    for (px,py),(dx,dy) in zip(pts,dirs):
        cv2.line(img,(px,py),(px+dx*corner_len,py),color,thickness+1)
        cv2.line(img,(px,py),(px,py+dy*corner_len),color,thickness+1)
    cv2.rectangle(img,(x,y),(x+w,y+h),color,1)

def draw_emotion_bars(img, scores, x, y, w):
    if not scores:
        return
    bar_h = 10
    bar_w = w
    emotions = ['happy','sad','angry','surprise','neutral']
    for i, emo in enumerate(emotions):
        score = scores.get(emo, 0)
        filled = int((score/100) * bar_w)
        by = y + i*(bar_h+3)
        cv2.rectangle(img, (x, by), (x+bar_w, by+bar_h), (50,50,50), -1)
        col = emotion_colors.get(emo, (200,200,200))
        cv2.rectangle(img, (x, by), (x+filled, by+bar_h), col, -1)
        cv2.putText(img, f"{emo[:3].upper()} {score:.0f}%",
                    (x+2, by+bar_h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1)

def draw_scan_line(img, y_pos):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.line(overlay, (0, y_pos), (w, y_pos), (0, 255, 0), 1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

def draw_right_panel(img, face_count, accepted, rejected, elapsed):
    H, W = img.shape[:2]
    panel_x = W - 160
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, 70), (W, 320), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, "SYSTEM STATUS", (panel_x+5, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    cv2.line(img, (panel_x+5, 95), (W-5, 95), (0,255,255), 1)

    items = [
        (f"RUNTIME: {int(elapsed)}s",   (200,200,200)),
        (f"FACES:   {face_count}",      (0,255,255)),
        (f"ACCEPTED:{accepted}",        (0,255,0)),
        (f"REJECTED:{rejected}",        (0,0,255)),
        ("",                            (0,0,0)),
        ("AGENTS ONLINE:",              (0,200,255)),
        (" Detection  [OK]",            (0,255,0)),
        (" Recognition[OK]",            (0,255,0)),
        (" Emotion    [OK]",            (0,255,0)),
        (" Liveness   [OK]",            (0,255,0)),
        (" Tracker    [OK]",            (0,255,0)),
        (" Orchestrate[OK]",            (0,255,0)),
    ]
    for i,(text,color) in enumerate(items):
        cv2.putText(img, text, (panel_x+5, 115+i*17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        fail_count += 1
        if fail_count > 30:
            break
        time.sleep(0.03)
        continue
    fail_count  = 0
    frame_count += 1

    # Scan line animation
    scan_line_y = (scan_line_y + 4) % frame.shape[0]
    draw_scan_line(frame, scan_line_y)

    if frame_count % PROCESS_EVERY == 0:
        try:
            output     = detection.run(frame)
            faces      = output['faces']
            results    = []
            identities = []
            for face_data in faces:
                result = orchestrator.orchestrate(face_data)
                results.append(result)
                identities.append(result['recognition']['identity'])
                total_detected += 1
                if 'ACCEPT' in result['decision']:
                    total_accepted += 1
                else:
                    total_rejected += 1
            track_map      = tracker.update(faces, identities)
            last_faces     = faces
            last_results   = results
            last_track_map = track_map
        except Exception as e:
            print(f"[WARN] {e}")

    try:
        face_count = len(last_faces)
        elapsed    = time.time() - start_time

        for i,(face_data,result) in enumerate(zip(last_faces,last_results)):
            raw_box       = face_data['box']
            decision      = result['decision']
            identity      = result['recognition']['identity']
            emotion_label = result['emotion'].get('dominant_emotion','neutral')
            emotion_scores= result['emotion'].get('emotion_scores',{})
            score         = result['recognition']['score']
            source        = face_data.get('source','?')
            liveness_score= result['liveness'].get('liveness_score',0)

            track_id = None
            for tid,idx2 in last_track_map.items():
                if idx2 == i:
                    track_id = tid
                    break
            if track_id is None:
                track_id = i+1

            x,y,w,h = smooth_box(track_id, raw_box)

            if track_id not in smooth_labels or frame_count % 6 == 0:
                smooth_labels[track_id] = {
                    'identity':      identity,
                    'decision':      decision,
                    'score':         score,
                    'emotion_label': emotion_label,
                    'emotion_scores':emotion_scores,
                    'source':        source,
                    'liveness':      liveness_score,
                }
            lbl = smooth_labels[track_id]

            if 'ACCEPT' in lbl['decision']:
                color    = (0,255,0)
                bg_color = (0,120,0)
            elif 'ESCALATE' in lbl['decision']:
                color    = (0,165,255)
                bg_color = (0,80,140)
            else:
                color    = (0,0,255)
                bg_color = (130,0,0)

            emo_color = emotion_colors.get(lbl['emotion_label'], (200,200,200))

            # Fancy corner box
            draw_corner_box(frame, x, y, w, h, color)

            # Pulsing circle at face center
            cx,cy = x+w//2, y+h//2
            pulse_r = 5 + int(3*np.sin(frame_count*0.2))
            cv2.circle(frame, (cx,cy), pulse_r, color, 1)

            # Top info bar
            top_y = max(50, y)
            cv2.rectangle(frame,(x,top_y-50),(x+w,top_y), bg_color,-1)
            cv2.putText(frame, f"ID#{track_id} {lbl['identity']}",
                        (x+4,top_y-34), cv2.FONT_HERSHEY_SIMPLEX, 0.52,(255,255,255),2)
            cv2.putText(frame, f"{lbl['decision']}  {lbl['score']:.0%}",
                        (x+4,top_y-18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,(255,255,255),1)
            cv2.putText(frame, f"LIVE:{lbl['liveness']:.0%} | {lbl['source']}",
                        (x+4,top_y-4),  cv2.FONT_HERSHEY_SIMPLEX, 0.35,(180,180,180),1)

            # Bottom emotion bar with color
            bot_y = min(frame.shape[0]-70, y+h)
            cv2.rectangle(frame,(x,bot_y),(x+w,bot_y+20), emo_color,-1)
            cv2.putText(frame, emoji_map.get(lbl['emotion_label'],'NEUTRAL'),
                        (x+4,bot_y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0),2)

            # Emotion mini bars
            draw_emotion_bars(frame, lbl['emotion_scores'], x, bot_y+24, w)

            if lbl['identity']=='UNKNOWN_PERSON' and tracker.get_alert_status(track_id,lbl['identity']):
                alert_display.append({
                    'msg': f"⚠ SECURITY ALERT! Unknown ID#{track_id} Detected!",
                    'time': time.time()
                })

        # ── TOP HUD ──────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(frame.shape[1],62),(10,10,10),-1)
        cv2.addWeighted(overlay,0.75,frame,0.25,0,frame)
        cv2.putText(frame,"MULTI-AGENT FACE RECOGNITION SYSTEM",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.putText(frame,
                    f"Faces:{face_count}  Tracked:{len(tracker.tracked_faces)}  Accepted:{total_accepted}  Frame:{frame_count}",
                    (10,48), cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,180),1)

        # ── RIGHT PANEL ───────────────────────────────────────────
        draw_right_panel(frame, face_count, total_accepted, total_rejected, elapsed)

        # ── ALERTS ───────────────────────────────────────────────
        now = time.time()
        alert_display = [a for a in alert_display if now-a['time']<4.0]
        for j,alert in enumerate(alert_display[-3:]):
            ay = 80 + j*36
            cv2.rectangle(frame,(0,ay-24),(580,ay+8),(0,0,180),-1)
            cv2.putText(frame, alert['msg'],(10,ay),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    except Exception as e:
        print(f"[WARN] Draw: {e}")

    cv2.imshow('Multi-Agent Face Recognition System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
with open('audit_log.json','w') as f:
    json.dump(orchestrator.audit_log,f,indent=2)
with open('tracker_log.json','w') as f:
    json.dump(tracker.alert_log,f,indent=2)
print("[*] Done! Logs saved.")