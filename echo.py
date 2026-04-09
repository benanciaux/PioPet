from unitree_sdk2py.go2.robot_interface import RobotInterface

robot = RobotInterface()

def on_speech(text):
    print("Heard:", text)
    
    # Send it back to speaker
    robot.audio.tts(text)

# Hook into robot's speech system
robot.audio.set_speech_callback(on_speech)

# Keep running
while True:
    pass