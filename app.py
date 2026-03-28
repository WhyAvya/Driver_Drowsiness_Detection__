import cv2
import time

from src.camera import Camera
from src.mediapipe_eye_detector import MediaPipeEyeDetector
from src.inference import EyeStateInference
from src.drowsiness_logic import DrowsinessLogic
from src.alarm import Alarm


MODEL_PATH = "models/best_eye_state_effnet.pth"
ALARM_SOUND = "assets/alarm.wav"


def draw_text(frame, text, x, y, scale=0.7, color=(0, 255, 0), thickness=2):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def run():
    cam = Camera(index=0, width=1280, height=720)
    detector = MediaPipeEyeDetector()
    inference = EyeStateInference(model_path=MODEL_PATH)
    logic = DrowsinessLogic(
        closed_on_threshold=0.65,
        open_off_threshold=0.40,
        frame_threshold=15,
        window_size=8,
        ear_threshold=0.23,
        ear_weight=0.10,
    )
    alarm = Alarm(sound_path=ALARM_SOUND)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            eye_data = detector.get_eye_data(frame)

            if eye_data is None:
                logic.reset()
                alarm.stop()
                draw_text(frame, "No face detected", 30, 40, scale=1.0, color=(0, 0, 255))
                cv2.imshow("Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            left_result = inference.predict(eye_data["left_crop"])
            right_result = inference.predict(eye_data["right_crop"])

            result = logic.update(left_result, right_result, ear=eye_data["ear"])

            if result["is_drowsy"]:
                alarm.start()
                status_color = (0, 0, 255)
                status_text = "DROWSY"
            else:
                alarm.stop()
                status_color = (0, 255, 0)
                status_text = "AWAKE"

            # Draw eye bounding boxes
            if eye_data["left_bbox"] is not None:
                x1, y1, x2, y2 = eye_data["left_bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

            if eye_data["right_bbox"] is not None:
                x1, y1, x2, y2 = eye_data["right_bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

            # Draw labels
            if left_result is not None:
                draw_text(
                    frame,
                    f"Left:  {left_result['pred_label']}  {left_result['confidence']:.2f}",
                    30,
                    40,
                    color=(255, 255, 0),
                )
            else:
                draw_text(frame, "Left:  None", 30, 40, color=(0, 0, 255))

            if right_result is not None:
                draw_text(
                    frame,
                    f"Right: {right_result['pred_label']}  {right_result['confidence']:.2f}",
                    30,
                    75,
                    color=(255, 255, 0),
                )
            else:
                draw_text(frame, "Right: None", 30, 75, color=(0, 0, 255))

            draw_text(frame, f"EAR: {eye_data['ear']:.3f}", 30, 110, color=(0, 255, 255))
            draw_text(frame, f"Score: {result['score']:.3f}", 30, 145, color=(0, 255, 255))
            draw_text(frame, f"Counter: {result['counter']}/{logic.frame_threshold}", 30, 180, color=(0, 255, 255))
            draw_text(frame, f"Status: {status_text}", 30, 220, scale=1.0, color=status_color)

            # FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            draw_text(frame, f"FPS: {fps:.1f}", 30, 255, color=(200, 200, 200))

            draw_text(frame, "Press Q to quit", 30, 290, color=(180, 180, 180))

            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        alarm.stop()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()