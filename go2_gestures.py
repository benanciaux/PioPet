import threading

import cv2
import numpy as np
import time, math, sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    FINGERS = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }

    def __init__(self, max_hands=2, mode="live",
                 model_path="handmark_task"):
        self.latest_landmarks = None
        self.lock = threading.Lock()

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode = (vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.LIVE_STREAM),
            num_hands = max_hands,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
            result_callback = self._on_result if mode == "live" else None,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.hands = vision.HandLandmarksConnections
        self.draw = vision.drawing_utils
        self.image = None

    def _on_result(self, result, image, timestamp_ms):
        if not result.hand_landmarks:
            return

        with self.lock:
            self.latest_landmarks = result.hand_landmarks[0]

    def find_hands(self, image, draw=True):
        self.image = image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        self.landmarker.detect_async(mp_image, timestamp_ms)

        if draw and self.latest_landmarks is not None:
            self.draw.draw_landmarks(image, self.latest_landmarks, self.hands.HAND_CONNECTIONS)

    def get_landmarks(self):
        with self.lock:
            lm = self.latest_landmarks
        if lm is None:
            return None
        h, w, _ = self.image.shape
        return np.array([[i, int(p.x*w), int(p.y*h), float(p.z)] for i, p in enumerate(lm)])

    def get_angle(self, lms, p1, p2, p3):
        ax, ay, az = lms[p2][1] - lms[p1][1], lms[p2][2] - lms[p1][2], lms[p2][3] - lms[p1][3]
        cx, cy, cz = lms[p3][1] - lms[p2][1], lms[p3][2] - lms[p2][2], lms[p3][3] - lms[p2][3]

        dot = ax*cx + ay*cy + az*cz
        sin_t = math.sqrt((ay*cz-az*cy)**2 + (az*cx-ax*cz)**2 + (ax*cy-ay*cx)**2)
        return math.degrees(math.atan2(sin_t, dot))

    def get_distance(self, p1, p2, img=None):
        _, x1, y1, z1 = p1
        _, x2, y2, z2 = p2
        cx, cy = int(x1 + x2) // 2, int(y1 + y2) // 2

        if img is not None:
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), max(1, 5 // 3))
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2

        return math.sqrt(dx**2 + dy**2 + dz**2)

    def finger_state(self, lms, name):
        mcp, pip, dip, tip = self.FINGERS[name]
        ang1 = self.get_angle(lms, mcp, pip, dip)
        ang2 = self.get_angle(lms, pip, dip, tip)
        ang3 = self.get_angle(lms, mcp, dip, tip)
        total = ang1 + ang2 + ang3

        if name == "thumb":
            if total < 60:
                return "extended"
            tx = lms[tip][1] - lms[0][1]
            ty = lms[tip][2] - lms[0][2]
            tz = lms[tip][3] - lms[0][3]
            px = lms[5][1] - lms[0][1]
            py = lms[5][2] - lms[0][2]
            pz = lms[5][3] - lms[0][3]
            dot = tx*px + ty*py + tz*pz
            mag = px*px + py*py + pz*pz + 1e-6
            if total > 120 and (dot/mag) < 0.7:
                return "folded"
            return "curled"

        if total < 90: return "extended"
        if total > 120 and lms[tip][2] > lms[pip][2]: return "folded"
        return "curled"

    def get_finger_states(self, lms):
        return {name: self.finger_state(lms, name) for name in self.FINGERS}

    def count_fingers(self, lms, states=None):
        if states is None:
            states = self.get_finger_states(lms)
        extended = sum(1 for s in states.values() if s == "extended")
        folded = 5 - extended
        return extended, folded

    def get_hand_direction(self, lms, states=None, thresh=0.1):
        if states is None:
            states = self.get_finger_states(lms)

        if states["thumb"] == "extended" and all(
            states[f] == "folded" for f in ["index", "middle", "ring", "pinky"]):
            dx = lms[4][1] - lms[2][1]
            dy = lms[4][2] - lms[2][2]
        else:
            dx = lms[9][1] - lms[0][1]
            dy = lms[9][2] - lms[0][2]

        if abs(dx) < thresh and abs(dy) < thresh:
            return "center"
        if abs(dy) >= abs(dx):
            return "up" if dy < 0 else "down"
        else:
            return "right" if dx > 0 else "left"

    def palm_normal(self, lms):
        ax = lms[5][1]-lms[0][1]
        ay = lms[5][2]-lms[0][2]
        az = lms[5][3]-lms[0][3]
        bx = lms[17][1]-lms[0][1]
        by = lms[17][2]-lms[0][2]
        bz = lms[17][3]-lms[0][3]
        nx = ay*bz - az*by
        ny = az*bx - ax*bz
        nz = ax*by - ay*bx
        mag = math.sqrt(nx*nx + ny*ny + nz*nz) + 1e-6
        return nx/mag, ny/mag, nz/mag

    def get_palm_orientation(self, lms):
        normal = self.palm_normal(lms)
        x, y, z = normal

        threshold_up = 0.3
        threshold_flat = 0.2
        threshold_forward = 0.2

        if y > threshold_up:
            return 'up'
        elif y < -threshold_up:
            return 'down'

        # check forward/backward (camera-relative)
        if z < -threshold_forward:
            return 'forward'
        elif z > threshold_forward:
            return 'backward'

        # check sideways / flat
        if abs(y) < threshold_flat:
            if abs(x) > abs(z):
                return 'sideways'
            else:
                return 'flat'
        return 'flat'

    def classify_gesture(self, lms, client=None):
        states = self.get_finger_states(lms)
        extended, folded = self.count_fingers(lms, states)
        direction = self.get_hand_direction(lms, states)
        palm_orientation = self.get_palm_orientation(lms)

        # FIST
        # print("palm_orientation", palm_orientation, "direction", direction)
        # print(states["thumb"], states["index"], states["middle"], states["ring"], states["pinky"])

        # stop
        if all(
            states[f] == "extended" for f in ["thumb", "index", "middle", "ring", "pinky"]
        ):
            if palm_orientation == "forward" and direction == "up":
                return "COME"
            if palm_orientation == "backward" and direction == "up":
                return "STOP"
            if palm_orientation == "down" and direction == "center":
                return "STAND_DOWN"
        elif states["thumb"] == "extended" and all(
            (states[f] == "folded" or states[f] == "curled") for f in ["index", "middle", "ring", "pinky"]
        ):
            if direction == "left":
                if palm_orientation == "forward":
                    return "STAND_UP"
                elif palm_orientation == "backward":
                    return "STAND_DOWN"
        elif states["index"] == "extended":
            if (states["thumb"] == "folded" or states["thumb"] == "curled") and all(
            states[f] == "folded" for f in ["middle", "ring", "pinky"]):
                if direction == "left" and palm_orientation == "forward":
                    return "POINTING_LEFT"
                elif direction == "right" and palm_orientation == "backward":
                    return "POINTING_RIGHT"
        elif folded == 5:
            if palm_orientation == "forward" or palm_orientation == "backward":
                return "SIT"

        return "No action"

    def close(self):
        self.landmarker.close()


def stable_gesture(gesture, buffer=[], size=10, threshold=7):
    buffer.append(gesture)
    if len(buffer) > size:
        buffer.pop(0)

    counts = {}
    for g in buffer:
        counts[g] = counts.get(g, 0) + 1

    best = max(counts, key=counts.get)
    return best if counts[best] >= threshold else "No action"

def execute_command(gesture, client=None):
    if gesture == "COME":
        if client: client.Move(0, 0, 0)
        # print("COME")
    elif gesture == "STOP":
        if client: client.StopMove()
        # print("STOP")
    elif gesture == "STAND_UP":
        if client: client.StandUp()
        # print("STAND_UP")
    elif gesture == "STAND_DOWN":
        if client: client.StandDown()
        # print("STAND_DOWN")
    elif gesture == "POINTING_LEFT":
        if client: client.Move(0, 0.5, 0)
        # print("POINTING_LEFT")
    elif gesture == "POINTING_RIGHT":
        if client: client.Move(0, -0.5, 0)
        # print("POINTING_RIGHT")
    elif gesture == "SIT":
        if client: client.Sit()
        # print("SIT")
    elif gesture == "No action":
        if client: client.StopMove()
        # print("No action")

no_robot = True

if __name__ == "__main__":
    if no_robot:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        if len(sys.argv)>1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

        video_client = VideoClient()  # Create a video client
        video_client.SetTimeout(3.0)
        video_client.Init()

        sports_client = SportClient()
        sports_client.SetTimeout(10.0)
        sports_client.Init()

        code, data = video_client.GetImageSample()

        detector = HandDetector(model_path='hand_landmarker.task')
        last_gesture = None

        while code == 0:
            code, data = video_client.GetImageSample()

            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            image = cv2.flip(image, 1)
            if image is None: continue
            detector.find_hands(image, draw=True)
            landmarks = detector.get_landmarks()

            if landmarks is not None:
                raw_gesture = detector.classify_gesture(landmarks)
                gesture = stable_gesture(raw_gesture)

                if gesture != last_gesture:
                    print(gesture)
                    last_gesture = gesture

                execute_command(gesture, sports_client)

            cv2.imshow("front_camera", image)
            if cv2.waitKey(20) == 27:
                break

        if code != 0:
            print("Get image sample error. code:", code)
        else:
            cv2.imwrite("front_image.jpg", image)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        detector = HandDetector(model_path=r'C:\Users\matos\PycharmProjects\UGE\models\hand_landmarker.task')
        last_gesture = None

        while True:
            success, image = cap.read()
            if success is None or image is None: continue
            image = cv2.flip(image, 1)

            detector.find_hands(image, draw=True)
            landmarks = detector.get_landmarks()

            if landmarks is not None:
                raw_gesture = detector.classify_gesture(landmarks)
                gesture = stable_gesture(raw_gesture)

                if gesture != last_gesture:
                    print(gesture)
                    last_gesture = gesture

                execute_command(gesture)


            cv2.imshow("front_camera", image)
            if cv2.waitKey(20) == 27:
                break
        cap.release()
        cv2.destroyWindow("front_camera")




