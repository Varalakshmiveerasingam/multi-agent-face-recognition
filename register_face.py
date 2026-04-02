import cv2
import sys
import time
sys.path.append('.')
from agents.recognition_agent import FaceRecognitionAgent

agent = FaceRecognitionAgent()
name = input("Enter your name: ")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

print("[*] Warming up camera... please wait 5 seconds")
time.sleep(2)
for _ in range(30):
    cap.read()
    time.sleep(0.1)
print("[*] Camera ready! Press SPACE to capture, Q to quit")

sample_count = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        time.sleep(0.05)
        continue

    display = frame.copy()

    # Progress bar
    bar_width = int((sample_count / 6) * frame.shape[1])
    cv2.rectangle(display, (0, frame.shape[0]-20), (bar_width, frame.shape[0]), (0,255,0), -1)

    cv2.putText(display, f"Name: {name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(display, f"Samples: {sample_count}/6", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(display, "SPACE = capture | Q = quit", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Guide angles
    angles = ["1: Look STRAIGHT", "2: Look LEFT", "3: Look RIGHT",
              "4: Move CLOSER", "5: SMILE", "6: NEUTRAL face"]
    if sample_count < 6:
        cv2.putText(display, f"Now: {angles[sample_count]}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 2)

    cv2.imshow("Register Face - Press SPACE", display)

    key = cv2.waitKey(1)
    if key == 32:
        face = cv2.resize(frame, (224, 224))
        agent.register_face(name, face)
        sample_count += 1
        print(f"[+] Sample {sample_count}/6 saved! ({angles[sample_count-1]})")
        if sample_count >= 6:
            print(f"[DONE] All 6 samples registered for {name}!")
            break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[*] Registration complete! {sample_count} samples saved for {name}")