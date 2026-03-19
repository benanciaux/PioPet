import random
import speech_recognition as sr
import pyttsx3
import time

#go2
from unitree_sdk2py.go2.robot_interface import RobotInterface

#initalize robot
robot = RobotInterface()
robot.init()

#voice
engine = pyttsx3.init()

def speak(text):
    print("Robot:", text)
    engine.say(text)
    engine.runAndWait()

recognizer = sr.Recognizer()

#robot tricks
def stand():
    speak("Standing")
    robot.stand()

def sit():
    speak("Sitting")
    robot.sit()

def lay_down():
    speak("Laying down")
    robot.lie_down()

def shake():
    speak("Nice to meet you")
    robot.wave_hand()

def dance_1():
    speak("Imma dance")
    robot.dance1()

def dance_2():
    speak("I'm the best dog dancer!")
    robot.dance2()

def random_dance():
    random.choice([dance_1, dance_2])()

#command matching
COMMANDS = {
    "sit": sit,
    "stand": stand,
    "lay down": lay_down,
    "lie down": lay_down,
    "shake": shake,
    "dance": random_dance,
    "do a dance": random_dance,
}

#speech listener
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, phrase_time_limit=4)
    try:
        text = recognizer.recognize_google(audio).lower()
        print("Heard:", text)
        return text
    except:
        return ""

#command handler
def handle_command(text):

    for phrase, action in COMMANDS.items():
        if phrase in text:
            action()
            return True
    if "stop listening" in text:
        speak("Okay goodbye :(")
        return False
    speak("I don't understand")
    return True

#main loop
speak("Voice tricks activated")

running = True

while running:
    command = listen()
    if command:
        running = handle_command(command)
    time.sleep(0.2)
