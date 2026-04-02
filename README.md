@"
# Multi-Agent Intelligent Face Recognition System

A real-time face recognition system built with 5 specialized AI agents using Deep Learning.

## Agents
- Agent 1: Face Detection (MTCNN)
- Agent 2: Face Recognition (ArcFace + DeepFace)
- Agent 3: Emotion Analysis (7 emotions)
- Agent 4: Liveness Detection (Anti-Spoofing)
- Agent 5: Orchestrator (Decision Engine)

## Features
- Real-time face detection and recognition
- Emotion detection (happy, sad, neutral, angry, fear, surprise, disgust)
- Anti-spoofing liveness check
- Multi-face detection simultaneously
- Decision: ACCEPTED / ESCALATE / REJECTED

## Tech Stack
Python | OpenCV | TensorFlow | DeepFace | ArcFace | MTCNN

## How to Run
1. pip install deepface opencv-python mtcnn tensorflow fer numpy pillow tf-keras
2. python register_face.py
3. python main.py
4. Press Q to quit
"@ | Out-File -FilePath README.md -Encoding utf8
