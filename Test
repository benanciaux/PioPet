import speech_recognition as sr                                 #for speech_recognition
import time
from unitree_sdk2py.go2.robot_interface import RobotInterface

from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


robot = RobotInterface()
robot.init()

#speech Recognition
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen():
    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print("Heard:", command)
        return command
    except:
        return ""

audio_client.TtsMaker("BEN BEN HERE, BOW DOWN TO MY POWER",0)

time.sleep(3)

while True:
    command = listen()
    if command:
        audio_client.TtsMaker(command,0)
