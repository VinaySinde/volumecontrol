import cv2
import time
import math
import os
import urllib.request

import mediapipe as mp
from pycaw.pycaw import AudioUtilities


# ============================================================
# MEDIAPIPE HAND MODEL
# ============================================================

MODEL_FILE = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_FILE):
    print("Downloading hand model...")
    
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Hand model downloaded successfully.")
    except Exception as e:
        print("Could not download hand model.")
        print(e)
        input("Press Enter to exit...")
        exit()


# ============================================================
# WINDOWS LAPTOP VOLUME
# ============================================================

try:
    device = AudioUtilities.GetSpeakers()
    volume = device.EndpointVolume

except Exception as e:
    print("Could not access laptop volume.")
    print(e)
    input("Press Enter to exit...")
    exit()


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA TEST / DETECTION
# ============================================================

print("Starting camera test...")

camera = None
camera_index = -1

for index in [0, 1, 2]:

    print(f"Trying camera {index}...")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if cap.isOpened():

        time.sleep(1)

        ret, frame = cap.read()

        if ret and frame is not None:

            camera = cap
            camera_index = index

            print(f"Camera {index} found successfully!")

            break

        cap.release()


if camera is None:

    print()
    print("ERROR: No working camera was found.")
    print()
    print("Check:")
    print("1. Windows Camera permission")
    print("2. Laptop camera privacy switch")
    print("3. Camera driver")
    print("4. Close Zoom/Teams/Camera app")
    print()

    input("Press Enter to exit...")
    exit()


print()
print(f"Using camera index: {camera_index}")
print()
print("==========================================")
print(" HAND GESTURE LAPTOP VOLUME CONTROL")
print("==========================================")
print()
print("Thumb + Index FAR   = Volume UP")
print("Thumb + Index CLOSE = Volume DOWN")
print()
print("Press Q to quit.")
print()


# ============================================================
# HAND CONNECTIONS
# ============================================================

CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# VOLUME SETTINGS
# ============================================================

MIN_DISTANCE = 30
MAX_DISTANCE = 250

last_volume = -1
timestamp = 0


# ============================================================
# FUNCTION: SET LAPTOP VOLUME
# ============================================================

def set_volume(percent):

    percent = max(0, min(100, percent))

    min_db, max_db, _ = volume.GetVolumeRange()

    volume_db = min_db + (
        percent / 100.0
    ) * (max_db - min_db)

    volume.SetMasterVolumeLevel(
        volume_db,
        None
    )


# ============================================================
# FUNCTION: DRAW HAND
# ============================================================

def draw_hand(frame, landmarks):

    height, width, _ = frame.shape

    points = []

    # Draw points
    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    # Draw hand lines
    for start, end in CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2
        )

    return points


# ============================================================
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = camera.read()

        if not ret:

            print("ERROR: Cannot read frame from camera.")
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape


        # ====================================================
        # CONVERT CAMERA IMAGE
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ====================================================
        # HAND DETECTION
        # ====================================================

        timestamp += 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # ====================================================
        # IF HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            points = draw_hand(
                frame,
                landmarks
            )


            # ------------------------------------------------
            # THUMB TIP = LANDMARK 4
            # INDEX TIP = LANDMARK 8
            # ------------------------------------------------

            thumb_x, thumb_y = points[4]

            index_x, index_y = points[8]


            # ------------------------------------------------
            # DISTANCE BETWEEN FINGERS
            # ------------------------------------------------

            distance = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )


            # Draw line between fingers

            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (0, 255, 255),
                4
            )


            # Draw circles

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (255, 0, 0),
                -1
            )


            # =================================================
            # CONVERT DISTANCE TO VOLUME
            # =================================================

            volume_percent = int(
                (
                    (distance - MIN_DISTANCE)
                    /
                    (MAX_DISTANCE - MIN_DISTANCE)
                ) * 100
            )


            # Keep between 0 and 100

            volume_percent = max(
                0,
                min(100, volume_percent)
            )


            # =================================================
            # CHANGE ACTUAL LAPTOP VOLUME
            # =================================================

            if abs(volume_percent - last_volume) >= 2:

                set_volume(volume_percent)

                last_volume = volume_percent


            # =================================================
            # DISPLAY DISTANCE
            # =================================================

            cv2.putText(
                frame,
                f"Finger Distance: {int(distance)} px",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


            # =================================================
            # DISPLAY VOLUME
            # =================================================

            cv2.putText(
                frame,
                f"Laptop Volume: {volume_percent}%",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )


            # =================================================
            # VOLUME BAR
            # =================================================

            bar_x = 30
            bar_y = 135
            bar_width = 350
            bar_height = 35


            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + bar_width,
                    bar_y + bar_height
                ),
                (255, 255, 255),
                2
            )


            filled_width = int(
                bar_width * volume_percent / 100
            )


            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + filled_width,
                    bar_y + bar_height
                ),
                (0, 255, 0),
                -1
            )


            # =================================================
            # STATUS
            # =================================================

            if distance > 150:

                status = "VOLUME INCREASING"

            elif distance < 70:

                status = "VOLUME DECREASING"

            else:

                status = "VOLUME CONTROL"


            cv2.putText(
                frame,
                status,
                (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            frame,
            "Move fingers apart = Volume UP",
            (30, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Move fingers together = Volume DOWN",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Laptop Hand Gesture Volume Control",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# ============================================================
# CLOSE
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Camera closed.")
print("Volume control program finished.")