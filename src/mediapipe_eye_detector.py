import cv2
import mediapipe as mp
import numpy as np


class MediaPipeEyeDetector:
    """
    Uses MediaPipe Face Mesh to detect eye landmarks, compute EAR,
    and crop left/right eyes for CNN classification.
    """

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _compute_ear(self, eye_points):
        # eye_points = 6 points
        A = self._euclidean(eye_points[1], eye_points[5])
        B = self._euclidean(eye_points[2], eye_points[4])
        C = self._euclidean(eye_points[0], eye_points[3])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    def _get_eye_points(self, landmarks, indices, w, h):
        return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

    def _extract_bbox_and_crop(self, frame, eye_points, pad_x_ratio=0.35, pad_y_ratio=0.60):
        xs = [p[0] for p in eye_points]
        ys = [p[1] for p in eye_points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = max(x_max - x_min, 1)
        height = max(y_max - y_min, 1)

        pad_x = int(width * pad_x_ratio)
        pad_y = int(height * pad_y_ratio)

        x1 = max(x_min - pad_x, 0)
        y1 = max(y_min - pad_y, 0)
        x2 = min(x_max + pad_x, frame.shape[1])
        y2 = min(y_max + pad_y, frame.shape[0])

        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = frame[y1:y2, x1:x2]
        bbox = (x1, y1, x2, y2)
        return crop, bbox

    def get_eye_data(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        left_points = self._get_eye_points(landmarks, self.LEFT_EYE, w, h)
        right_points = self._get_eye_points(landmarks, self.RIGHT_EYE, w, h)

        left_crop, left_bbox = self._extract_bbox_and_crop(frame, left_points)
        right_crop, right_bbox = self._extract_bbox_and_crop(frame, right_points)

        left_ear = self._compute_ear(left_points)
        right_ear = self._compute_ear(right_points)
        avg_ear = (left_ear + right_ear) / 2.0

        return {
            "left_points": left_points,
            "right_points": right_points,
            "left_crop": left_crop,
            "right_crop": right_crop,
            "left_bbox": left_bbox,
            "right_bbox": right_bbox,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "ear": avg_ear,
        }