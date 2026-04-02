from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class OrchestratorAgent:
    def __init__(self, recognition_agent, emotion_agent, liveness_agent):
        self.recognition_agent = recognition_agent
        self.emotion_agent = emotion_agent
        self.liveness_agent = liveness_agent
        self.audit_log = []

    def orchestrate(self, face_data):
        recog = self.recognition_agent.run(face_data)
        with ThreadPoolExecutor(max_workers=2) as ex:
            ef = ex.submit(self.emotion_agent.run, face_data)
            lf = ex.submit(self.liveness_agent.run, face_data)
            emotion = ef.result()
            liveness = lf.result()

        if not liveness['is_live']:
            decision = 'REJECTED_SPOOF'
            final_score = 0.0
        else:
            final_score = round(0.65*recog['score'] + 0.35*liveness['liveness_score'], 4)
            if final_score > 0.65:
                decision = 'ACCEPTED'
            elif final_score > 0.45:
                decision = 'ESCALATE'
            else:
                decision = 'REJECTED'

        entry = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'final_score': final_score,
            'recognition': recog,
            'emotion': emotion,
            'liveness': liveness
        }
        self.audit_log.append(entry)
        return entry
