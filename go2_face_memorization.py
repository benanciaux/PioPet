import sys
import os
import pickle
import random
import time
import re
import numpy as np
import cv2
import face_recognition

#add SDK path if needed
sys.path.append("./unitree_sdk2_python")

from unitree_sdk2py.go2.robot_interface import RobotInterface

MEMORY_FILE = "piopet_memory.pkl"

#connect to robot
robot = RobotInterface()

#responses
greeting_responses = [
    "Hi {name}! I'm PioPet!",
    "Nice to see you {name}!",
    "Hello {name}, welcome!"
]

recognition_responses = [
    "Hi {name}! Welcome back!",
    "Greetings {name}!",
    "Oh hi {name}!",
    "I see you {name}!"
]

#TTS
def speak(text):
    print("PioPet:", text)
    robot.audio.tts(text)   # <- uses robot speaker

#memory
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "rb") as f:
        known_faces, known_names = pickle.load(f)
else:
    known_faces = []
    known_names = []

def save_memory():
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump((known_faces, known_names), f)

#name extraction
def extract_name(text):
    match = re.search(r"(hi|hey) piopet i'm (\w+)", text)
    return match.group(2).capitalize() if match else None

#get camera
def get_frame():
    #SDK camera frame
    frame = robot.camera.get_frame()

    if frame is None:
        return None

    #convert to OpenCV
    frame = np.array(frame, dtype=np.uint8)
    return frame

#face capture
def capture_face():
    while True:
        frame = get_frame()

        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)

        if faces:
            encodings = face_recognition.face_encodings(rgb, faces)
            return encodings[0]

        cv2.imshow("Look at PioPet", frame)

        if cv2.waitKey(1) == 27:
            break

    return None

#face recognition
def recognize_face(face_encoding):
    if not known_faces:
        return None

    matches = face_recognition.compare_faces(
        known_faces,
        face_encoding,
        tolerance=0.5
    )

    distances = face_recognition.face_distance(
        known_faces,
        face_encoding
    )

    best_index = np.argmin(distances)

    if matches[best_index]:
        return known_names[best_index]

    return None

#vision loop
last_seen = {}

def watch_for_people():
    while True:
        frame = get_frame()

        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        for encoding in encodings:
            name = recognize_face(encoding)

            if name:
                now = time.time()

                if name not in last_seen or now - last_seen[name] > 10:
                    speak(random.choice(recognition_responses).format(name=name))
                    last_seen[name] = now

        cv2.imshow("PioPet Vision", frame)

        if cv2.waitKey(1) == 27:
            break

#main
speak("PioPet is ready.")

while True:
    #no speech recognition then manual test input
    text = input("Type what you said: ").lower()

    name = extract_name(text)

    if name:
        speak(f"Nice to meet you {name}")

        encoding = capture_face()

        if encoding is not None:
            known_faces.append(encoding)
            known_names.append(name)
            save_memory()

            speak(random.choice(greeting_responses).format(name=name))

    watch_for_people()
