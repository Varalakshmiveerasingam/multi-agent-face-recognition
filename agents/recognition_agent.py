from deepface import DeepFace
import numpy as np
import pickle, os

class FaceRecognitionAgent:
    def __init__(self, db_path='database/face_db.pkl'):
        self.db_path = db_path
        self.threshold = 0.30
        self.database = self._load_database()

    def _load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def get_embedding(self, face_roi):
        result = DeepFace.represent(
            img_path=face_roi,
            model_name='ArcFace',
            enforce_detection=False
        )
        return np.array(result[0]['embedding'])

    def cosine_similarity(self, a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def recognize(self, face_roi):
        best_match, best_score = 'UNKNOWN_PERSON', 0.0
        try:
            query_emb = self.get_embedding(face_roi)
            for name, stored_embs in self.database.items():
                scores = [self.cosine_similarity(query_emb, e) for e in stored_embs]
                avg_score = float(np.mean(sorted(scores)[-3:]))
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = name
        except Exception as e:
            print(f"[RecognitionAgent] Error: {e}")

        if best_score > self.threshold:
            status = 'ACCEPTED'
        elif best_score > 0.22:
            status = 'ESCALATE'
            best_match = 'UNKNOWN_PERSON'
        else:
            status = 'REJECTED'
            best_match = 'UNKNOWN_PERSON'

        return {
            'agent': 'RecognitionAgent',
            'identity': best_match,
            'score': round(float(best_score), 4),
            'status': status
        }

    def register_face(self, name, face_roi):
        embedding = self.get_embedding(face_roi)
        if name not in self.database:
            self.database[name] = []
        self.database[name].append(embedding.tolist())
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database, f)
        print(f'[+] Registered: {name} (total samples: {len(self.database[name])})')

    def run(self, face_data):
        return self.recognize(face_data['roi'])