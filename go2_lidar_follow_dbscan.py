import numpy as np
import time
import speech_recognition as sr    #speech to text
import pyttsx3                     #text to speech
from sklearn.cluster import DBSCAN #predicting person

#robot
from unitree_sdk2py.go2.robot_interface import RobotInterface
from unitree_sdk2py.go2.lidar import LidarClient

#initialize
robot = RobotInterface()
robot.init()

lidar = LidarClient()
lidar.init()

engine = pyttsx3.init()

DESIRED_DISTANCE = 1.2
TARGET_LOST_TIMEOUT = 3.0

following = False
last_seen_time = 0

#voice
def speak(text):
    print("Robot:", text)
    engine.say(text)
    engine.runAndWait()

recognizer = sr.Recognizer()

def heard_follow_command():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=3)

    try:
        text = recognizer.recognize_google(audio).lower()
        print("Heard:", text)
        return "follow me" in text
    except:
        return False

#lidar
def scan_to_points(scan):
    #convert polar LiDAR scan to XY points

    angles = (
        scan.angle_min +
        np.arange(len(scan.ranges)) * scan.angle_increment
    )

    ranges = np.array(scan.ranges)
    valid = np.isfinite(ranges) & (ranges > 0.05)

    ranges = ranges[valid]
    angles = angles[valid]

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    return np.vstack((x, y)).T

def cluster_points(points):
    #DBSCAN clustering - uses AI to predict human

    if len(points) < 20:
        return []

    clustering = DBSCAN(eps=0.25, min_samples=8).fit(points)
    labels = clustering.labels_

    clusters = []

    for label in set(labels):
        if label == -1:
            continue

        cluster = points[labels == label]
        clusters.append(cluster)

    return clusters

def detect_human_cluster(clusters):
    #select cluster most likely to be a human

    best_target = None
    best_distance = 999

    for cluster in clusters:

        center = np.mean(cluster, axis=0)
        distance = np.linalg.norm(center)

        #cluster width estimation
        width = np.max(cluster[:,1]) - np.min(cluster[:,1])

        #human-like filters
        if 0.2 < width < 0.9 and 0.4 < distance < 3.5:
            if distance < best_distance:
                best_distance = distance
                best_target = center

    return best_target

#follow
def follow_controller(target):

    x, y = target

    distance = np.sqrt(x*x + y*y)
    angle = np.arctan2(y, x)

    #proportional control
    forward = (distance - DESIRED_DISTANCE) * 0.6
    turn = angle * 1.2

    #clamp speeds
    forward = np.clip(forward, -0.2, 0.4)
    turn = np.clip(turn, -0.8, 0.8)

    return forward, turn

#main follow loop
def follow_loop():

    global last_seen_time

    speak("Following activated")

    while True:

        scan = lidar.get_scan()
        points = scan_to_points(scan)

        #focus front hemisphere
        points = points[points[:,0] > 0]

        clusters = cluster_points(points)
        target = detect_human_cluster(clusters)

        if target is not None:

            last_seen_time = time.time()

            forward, turn = follow_controller(target)
            robot.velocity_move(forward, 0, turn)

        else:
            #target lost handling
            if time.time() - last_seen_time > TARGET_LOST_TIMEOUT:
                robot.velocity_move(0,0,0)
                speak("Target lost")
                break
            else:
                robot.velocity_move(0,0,0.3)  #slow search rotate
              
#main
speak("Voice control ready")

while True:
    if heard_follow_command():
        follow_loop()
