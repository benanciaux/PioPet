import threading
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import math
import cv2
import pyrealsense2 as rs
import argparse

INTERFACE = "enp1s0f0"
USE_GO2 = False
COOLDOWN_SECONDS = 5
GLOBAL_COOLDOWN = 3.0
REQUIRED_FRAMES = 60
MAX_DISTANCE = 1.0  # meters (adjust this)

WAKE_GESTURE = "open_palm"
SLEEP_GESTURE = "fist"
WAKE_HOLD_TIME = 1.0
WAKE_COOLDOWN = 1.0
toggle_triggered = False

active = False
wake_time = 0
wake_start_ts = None
current_gesture = None
stable_frames = 0
gesture_start_time = 0
triggered = False

cooldown = {}
last_command_time = 0

latest_result = None
result_lock = threading.Lock()

latest_go2_frame = None
go2_frame_lock = threading.Lock()

use_pyrealsense = False

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="camera index")
args = parser.parse_args()

def connect_go2():
    try:
        import os
        os.environ["CYCLONEDDS_URI"] = (
            "<CycloneDDS><Domain><General>"
            f"<NetworkInterfaceAddress>{INTERFACE}</NetworkInterfaceAddress>"
            "</General></Domain></CycloneDDS>"
        )
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        ChannelFactoryInitialize(0, INTERFACE)
        go2 = SportClient()
        go2.SetTimeout(10.0)
        go2.Init()
        print("[Go2] Sport client connected")
        return go2
    except Exception as e:
        print(f"[Go2] Sport client not connected: {e}")
        return None

def connect_video():
    try:
        from unitree_sdk2py.go2.video.video_client import VideoClient
        video = VideoClient()
        video.SetTimeout(10.0)
        video.Init()
        print("[Go2] Video client connected")
        return video
    except Exception as e:
        print(f"[Go2] Video client not connected: {e}")
        return None

def go2_video_reader(video_client):
    """Continuously reads frames from Go2 camera and keeps only the latest."""
    global latest_go2_frame
    while True:
        try:
            code, data = video_client.GetImageSample()
            if code == 0 and data:
                arr = np.frombuffer(bytes(data), dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with go2_frame_lock:
                        latest_go2_frame = frame
        except Exception as e:
            print(f"[Video] Frame error: {e}")
            time.sleep(0.01)

FINGERS = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}

def on_result(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

def get_angle(lms, p1, p2, p3):
    ax = lms[p2].x - lms[p1].x
    ay = lms[p2].y - lms[p1].y
    az = lms[p2].z - lms[p1].z
    cx = lms[p3].x - lms[p2].x
    cy = lms[p3].y - lms[p2].y
    cz = lms[p3].z - lms[p2].z
    dot = ax*cx + ay*cy + az*cz
    sin_t = math.sqrt((ay*cz - az*cy)**2 + (az*cx - ax*cz)**2 + (ax*cy - ay*cx)**2)
    return math.degrees(math.atan2(sin_t, dot))

def finger_state(lms, name):
    mcp, pip, dip, tip = FINGERS[name]
    ang1 = get_angle(lms, mcp, pip, dip)
    ang2 = get_angle(lms, pip, dip, tip)
    ang3 = get_angle(lms, mcp, dip, tip)
    total = ang1 + ang2 + ang3

    if name == "thumb":
        if total < 60:
            return "extended"
        tx = lms[tip].x - lms[0].x
        ty = lms[tip].y - lms[0].y
        tz = lms[tip].z - lms[0].z
        px = lms[5].x - lms[0].x
        py = lms[5].y - lms[0].y
        pz = lms[5].z - lms[0].z
        dot = tx*px + ty*py + tz*pz
        mag = px*px + py*py + pz*pz + 1e-6
        if total > 120 and (dot / mag) < 0.7:
            return "folded"
        return "curled"

    if total < 90:
        return "extended"
    if total > 120 and lms[tip].y > lms[pip].y:
        return "folded"
    return "curled"

def get_thumb_state(lms):
    tip = lms[4]
    ip  = lms[3]
    return "up" if tip.y < ip.y else "down"

def get_fingers_states(lms):
    return {
        "thumb":  finger_state(lms, "thumb"),
        "index":  finger_state(lms, "index"),
        "middle": finger_state(lms, "middle"),
        "ring":   finger_state(lms, "ring"),
        "pinky":  finger_state(lms, "pinky"),
    }

def detect_gesture(hands):
    num_hands = len(hands)
    if num_hands == 2:
        ext_a = get_fingers_states(hands[0])
        ext_b = get_fingers_states(hands[1])
        four_fingers = ["index", "middle", "ring", "pinky"]

        if (all(ext_a[s] == "extended" for s in four_fingers) and
                all(ext_b[s] == "extended" for s in four_fingers)):
            return "two_open_palm"

        if (all(s in ("folded", "curled") for s in ext_a.values()) and
                all(s in ("folded", "curled") for s in ext_b.values())):
            return "two_fists"

        if (ext_a["thumb"] in ("folded", "curled") and
                ext_a["index"] == "extended" and
                ext_a["middle"] == "extended" and
                ext_a["ring"] in ("folded", "curled") and
                ext_a["pinky"] in ("folded", "curled") and
                ext_b["thumb"] in ("folded", "curled") and
                ext_b["index"] == "extended" and
                ext_b["middle"] == "extended" and
                ext_b["ring"] in ("folded", "curled") and
                ext_b["pinky"] in ("folded", "curled")):
            return "wallow"

        if (ext_a["index"] == "extended" and
                ext_a["middle"] in ("folded", "curled") and
                ext_a["ring"] in ("folded", "curled") and
                ext_a["pinky"] == "extended" and
                ext_b["index"] == "extended" and
                ext_b["middle"] in ("folded", "curled") and
                ext_b["ring"] in ("folded", "curled") and
                ext_b["pinky"] == "extended"):
            return "two_rock_sign"

        if (ext_a["pinky"] == "extended" and
                all(ext_a[s] in ("folded", "curled") for s in ["thumb", "index", "middle", "ring"]) and
                ext_b["pinky"] == "extended" and
                all(ext_b[s] in ("folded", "curled") for s in ["thumb", "index", "middle", "ring"])):
            return "both_pinky"
    else:
        ext = get_fingers_states(hands[0])
        t, i, m, r, p = ext.values()

        if all(s in ("folded", "curled") for s in ext.values()):
            return "fist"

        if all(s == "extended" for s in ext.values()):
            return "open_palm"

        if (t in ("folded", "curled") and i == "extended" and
                all(ext[s] == "folded" for s in ["middle", "ring", "pinky"])):
            return "index_up"

        if (t in ("folded", "curled") and m == "extended" and
                all(ext[s] == "folded" for s in ["index", "pinky", "ring"])):
            return "middle_up"

        if (t in ("folded", "curled") and r == "extended" and
                all(ext[s] == "folded" for s in ["index", "middle", "pinky"])):
            return "ring_up"

        if (t in ("folded", "curled") and p == "extended" and
                all(ext[s] == "folded" for s in ["index", "middle", "ring"])):
            return "pinky_up"

        if (t in ("folded", "curled") and i == "extended" and m == "extended" and
                r in ("folded", "curled") and p in ("folded", "curled")):
            return "peace"

        if i == "extended" and m in ("folded", "curled") and r in ("folded", "curled") and p == "extended":
            return "one_rock_sign"





        if (t in ("extended", "curled") and
                all(ext[s] in ("folded", "curled") for s in ["index", "middle", "ring", "pinky"])):
            thumb_state = get_thumb_state(hands[0])
            return "thumbs_up" if thumb_state == "up" else "thumbs_down"

    return None

def send_command(gesture, go2, is_active):
    if go2 is None:
        print(f"[GESTURE] {gesture} (no go2)")
        return
    if not is_active:
        print(f"[IGNORED] - SLEEP MODE] {gesture}")
        return
    actions = {
        "open_palm":        lambda: go2.StopMove(),
        "two_open_palms":   lambda: go2.Hello(),
        "two_fists":     lambda: go2.BalanceStand(),
        "one_rock_sign": lambda: go2.Dance1(),
        "two_rock_sign": lambda: go2.Dance2(),
        "pinky_up":        lambda: go2.Heart(),
        "peace":       lambda: go2.Content(),
        "index_up":         lambda: go2.Sit(),
        "thumbs_down": lambda: go2.StandDown(),
        "thumbs_up":   lambda: go2.StandUp(),
    }
    if gesture in actions:
        print(f"[GESTURE] Sending: {gesture}")
        actions[gesture]()

def draw_overlay(image, gesture, progress, is_triggered, is_active):
    status = "ACTIVE" if is_active else "INACTIVE"
    status_color = (0, 220, 0) if is_active else (0, 0, 255)
    cv2.putText(image, status, (470, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    if progress > 0 and is_active or (not is_active and gesture == WAKE_GESTURE):
        bar_x, bar_y = 20, 50
        bar_width = 300
        bar_height = 20

        filled_width = int(bar_width * progress)

        cv2.rectangle(image, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (50, 50, 50), -1)

        cv2.rectangle(image, (bar_x, bar_y),
                      (bar_x + filled_width, bar_y + bar_height),
                      (0, 255, 0), -1)

    cv2.putText(image, gesture or "", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

def get_hand_depth(depth_frame, landmark, w, h):
    x = int(landmark.x * w)
    y = int(landmark.y * h)

    # clamp
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    return depth_frame.get_distance(x, y)

def main():
    global triggered, toggle_triggered, current_gesture, gesture_start_time, last_command_time, stable_frames, active, wake_start_ts, wake_time

    # mediapipe setup
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=on_result,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # go2 setup
    go2 = connect_go2() if USE_GO2 else None
    # video = connect_video() if USE_GO2 else None
    video = None
    use_pyrealsense = True if args.camera == 1 else False

    if use_pyrealsense:
        # REALSENSE SETUP
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        align = rs.align(rs.stream.color)

    if args.camera:
        cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(1)

    if video is not None:
        # threading.Thread(target=go2_video_reader, args=(video,), daemon=True).start()
        print("[Video] Reader thread started — waiting for first frame...")
        # Wait up to 5 s for first frame
        for _ in range(50):
            with go2_frame_lock:
                if latest_go2_frame is not None:
                    break
            time.sleep(0.1)
        else:
            print("[Video] WARNING: no frame received yet, continuing anyway")
    else:
        print("[Video] Falling back to laptop webcam (index 0)")

    start_ms = int(time.time() * 1000)

    while True:
        if use_pyrealsense:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
        else:
            success, image = cap.read()
            if not success: continue

        image = cv2.resize(image, (640, 480))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000) - start_ms)

        with result_lock:
            result = latest_result

        gesture = None

        if use_pyrealsense:
            # depth filter - filtering out hands on the background
            hand_valid = False

            if result and result.hand_landmarks and depth_frame:
                h, w, _ = image.shape
                wrist = result.hand_landmarks[0][0]
                depth = get_hand_depth(depth_frame, wrist, w, h)

                if 0 < depth < MAX_DISTANCE:
                    hand_valid = True

            if result and result.hand_landmarks and hand_valid:
                gesture = detect_gesture(result.hand_landmarks)
        else:
            if result and result.hand_landmarks:
                gesture = detect_gesture(result.hand_landmarks)

        if gesture != current_gesture:
            current_gesture = gesture
            stable_frames = 0
            triggered = False
        elif gesture is not None:
            stable_frames += 1
        else:
            stable_frames = 0

        if gesture is not None and stable_frames >= REQUIRED_FRAMES and not triggered:
            now = time.time()
            if gesture == "open_palm" and not active:
                active = True
                triggered = True
                print("AWAKE")
            elif gesture == "fist" and active:
                active = False
                triggered = True
                print("SLEEP")
            elif active:
                last_sent = cooldown.get(gesture, 0)
                if ((
                    gesture == "open_palm" or now - last_sent >= COOLDOWN_SECONDS) and now - last_command_time >=
                        GLOBAL_COOLDOWN):
                    send_command(gesture, go2, active)
                    cooldown[gesture] = now
                    last_command_time = now
                    triggered = True

        progress = min(stable_frames / REQUIRED_FRAMES, 1.0) if gesture is not None else 0
        draw_overlay(image, gesture, progress, triggered, active)

        cv2.imshow("Go2 Gesture Control", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
