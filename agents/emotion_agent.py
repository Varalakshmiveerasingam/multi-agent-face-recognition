import cv2
import numpy as np

class EmotionAnalysisAgent:
    def __init__(self):
        self.use_deepface = False
        self.emotion_history = []
        self.history_size = 5
        self._load_model()

    def _load_model(self):
        try:
            from deepface import DeepFace
            self.use_deepface = True
            print("[EmotionAgent] Ready")
        except:
            print("[EmotionAgent] DeepFace not available")

    def analyze(self, face_roi):
        try:
            if self.use_deepface:
                from deepface import DeepFace

                # Brighten image before analysis
                lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                enhanced = cv2.merge((l,a,b))
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

                result = DeepFace.analyze(
                    img_path=enhanced,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                raw = result[0]['emotion']

                # Get face regions for correction
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face48 = cv2.resize(gray, (48, 48))
                mouth     = face48[30:45, 8:40]
                left_eye  = face48[10:22, 5:20]
                right_eye = face48[10:22, 28:43]
                l_brow    = face48[5:13,  4:20]
                r_brow    = face48[5:13,  28:44]
                nose      = face48[18:32, 15:33]
                cheeks    = face48[22:35, 5:43]

                mouth_var   = float(np.var(mouth))
                eye_var     = float((np.var(left_eye) + np.var(right_eye)) / 2)
                brow_var    = float((np.var(l_brow) + np.var(r_brow)) / 2)
                brightness  = float(np.mean(face48))
                mouth_open  = float(np.mean(mouth[3:8, 8:30]))
                nose_var    = float(np.var(nose))
                cheek_var   = float(np.var(cheeks))

                corrected = {}

                # HAPPY — needs real smile (high mouth variance)
                if mouth_var > 300:
                    corrected['happy'] = raw['happy'] * 2.0
                elif mouth_var > 180:
                    corrected['happy'] = raw['happy'] * 1.2
                else:
                    corrected['happy'] = raw['happy'] * 0.3

                # SAD — mouth drooping + low brightness
                if brightness < 100 and mouth_var < 150:
                    corrected['sad'] = raw['sad'] * 2.0
                elif mouth_open < 85 and mouth_var < 200:
                    corrected['sad'] = raw['sad'] * 1.4
                else:
                    corrected['sad'] = raw['sad'] * 0.7

                # ANGRY — furrowed brows + cheek tension
                if brow_var > 200 and cheek_var > 160:
                    corrected['angry'] = raw['angry'] * 2.2
                elif brow_var > 150:
                    corrected['angry'] = raw['angry'] * 1.5
                else:
                    corrected['angry'] = raw['angry'] * 0.6

                # SURPRISE — open mouth + wide eyes
                if mouth_open < 75 and eye_var > 220:
                    corrected['surprise'] = raw['surprise'] * 2.5
                elif eye_var > 180:
                    corrected['surprise'] = raw['surprise'] * 1.3
                else:
                    corrected['surprise'] = raw['surprise'] * 0.4

                # FEAR — wide eyes + dark + tense
                if eye_var > 230 and brightness < 115:
                    corrected['fear'] = raw['fear'] * 2.0
                else:
                    corrected['fear'] = raw['fear'] * 0.5

                # DISGUST — nose wrinkle + cheek tension
                if nose_var > 210 and cheek_var > 170:
                    corrected['disgust'] = raw['disgust'] * 2.0
                else:
                    corrected['disgust'] = raw['disgust'] * 0.6

                # NEUTRAL — wins when nothing else is strong
                top3 = sorted([corrected.get('happy',0), corrected.get('sad',0),
                               corrected.get('angry',0), corrected.get('surprise',0)],
                               reverse=True)
                if top3[0] < 25:
                    corrected['neutral'] = raw['neutral'] * 1.8
                else:
                    corrected['neutral'] = raw['neutral'] * 0.7

                # Normalize
                total = sum(corrected.values())
                if total > 0:
                    normalized = {k: round((v/total)*100, 2) for k, v in corrected.items()}
                else:
                    normalized = raw

                # Smooth over last 5 frames — stops flickering
                self.emotion_history.append(normalized)
                if len(self.emotion_history) > self.history_size:
                    self.emotion_history.pop(0)

                smoothed = {}
                for emo in normalized:
                    smoothed[emo] = round(
                        sum(h.get(emo, 0) for h in self.emotion_history) / len(self.emotion_history), 2
                    )

                dominant = max(smoothed, key=smoothed.get)
                confidence = round(smoothed[dominant], 2)

                return {
                    'agent': 'EmotionAgent',
                    'dominant_emotion': dominant,
                    'emotion_scores': smoothed,
                    'confidence': confidence
                }

        except Exception as e:
            print(f"[EmotionAgent] Error: {e}")

        return {
            'agent': 'EmotionAgent',
            'dominant_emotion': 'neutral',
            'emotion_scores': {},
            'confidence': 0.0
        }

    def run(self, face_data):
        return self.analyze(face_data['roi'])