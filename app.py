import cv2
from src.camera import Camera
from src.face_eye_detector import detect_face_and_eyes
from src.preprocess import preprocess_eye
from src.inference import EyeStateInference
from src.drowsiness_logic import DrowsinessDetector
from src.alarm import Alarm


def run():

    cam = Camera(0)

    engine = EyeStateInference(
    model_path="eye_state_mobilenet.pth",
    device=None
    )

    # ---- Drowsiness + Alarm ----
    drowsiness_detector = DrowsinessDetector(frame_threshold=30)
    alarm = Alarm("assets/alarm.wav")

    # ---- Memory for eye tracking ----
    prev_left_eye = None
    prev_right_eye = None

    try:
        while True:

            # ---- Capture frame ----
            frame = cam.get_frame()

            # ---- Detection ----
            face_box, left_eye, right_eye, debug = detect_face_and_eyes(
                frame, draw=True
            )

            cv2.imshow("Main Camera", debug)

            # =====================================================
            # 🔥 REUSE LAST EYES (CRITICAL FIX)
            # =====================================================
            if left_eye is None:
                left_eye = prev_left_eye
            else:
                prev_left_eye = left_eye

            if right_eye is None:
                right_eye = prev_right_eye
            else:
                prev_right_eye = right_eye

            # =====================================================
            # LEFT EYE
            # =====================================================
            if left_eye is not None:

                cv2.imshow("Left Eye", left_eye)

                eye_tensor = preprocess_eye(left_eye)

                if eye_tensor is not None:
                    label, confidence = engine.predict(eye_tensor)
                    left_closed = (label == 0) if label is not None else None

                    print(f"Left Eye: {label} ({confidence:.2f})")
                else:
                    left_closed = None

            else:
                left_closed = None
                cv2.imshow("Left Eye", frame)  # fallback


            # =====================================================
            # RIGHT EYE
            # =====================================================
            if right_eye is not None:

                cv2.imshow("Right Eye", right_eye)

                eye_tensor = preprocess_eye(right_eye)

                if eye_tensor is not None:
                    label, confidence = engine.predict(eye_tensor)
                    right_closed = (label == 0) if label is not None else None

                    print(f"Right Eye: {label} ({confidence:.2f})")
                else:
                    right_closed = None

            else:
                right_closed = None
                cv2.imshow("Right Eye", frame)  # fallback


            # =====================================================
            # DROWSINESS LOGIC
            # =====================================================
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