# app.py
import cv2
from src.camera import Camera
from src.face_eye_detector import detect_face_and_eyes
from src.preprocess import preprocess_eye
from src.inference import EyeStateInference
from src.drowsiness_logic import DrowsinessDetector
from src.alarm import Alarm


def run():
    cam = Camera(0)

    # ---- Layer 4: CNN Inference Engine ----
    engine = EyeStateInference(
        model_path="eye_state_mobilenet.pth",
        device="cpu"
    )

    # ---- Layer 5: Drowsiness + Alarm ----
    drowsiness_detector = DrowsinessDetector(frame_threshold=30)
    alarm = Alarm("assets/alarm.wav")

    try:
        while True:
            # ---- Layer 1: Capture frame ----
            frame = cam.get_frame()

            # ---- Layer 2: Face & eye detection ----
            face_box, left_eye, right_eye, debug = detect_face_and_eyes(
                frame, draw=True
            )

            cv2.imshow("Layer 2 Debug", debug)

            # ---------- LEFT EYE ----------
            if left_eye is not None:
                cv2.imshow("Left Eye", left_eye)
                eye_tensor = preprocess_eye(left_eye)
                label, confidence = engine.predict(eye_tensor)
                left_closed = (label == 0)
                print(f"Left Eye: {'CLOSED' if left_closed else 'OPEN'} ({confidence:.2f})")
            else:
                left_closed = None

            # ---------- RIGHT EYE ----------
            if right_eye is not None:
                cv2.imshow("Right Eye", right_eye)
                eye_tensor = preprocess_eye(right_eye)
                label, confidence = engine.predict(eye_tensor)
                right_closed = (label == 0)
                print(f"Right Eye: {'CLOSED' if right_closed else 'OPEN'} ({confidence:.2f})")
            else:
                right_closed = None

            # ---- Layer 5: Temporal reasoning ----
            is_drowsy = drowsiness_detector.update(left_closed, right_closed)

            if is_drowsy:
                print("🚨 DROWSINESS DETECTED 🚨")
                alarm.start()
            else:
                alarm.stop()

            # ---- Exit ----
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        alarm.stop()
        cam.release()
        cv2.destroyAllWindows()
