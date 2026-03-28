from collections import deque
import numpy as np


class DrowsinessLogic:
    """
    CNN primary:
    - Uses closed-eye probabilities from left/right eye classifier.
    - Adds a small EAR-based backup score.
    - Uses smoothing + consecutive-frame logic to avoid false positives.
    """

    def __init__(
        self,
        closed_on_threshold: float = 0.65,
        open_off_threshold: float = 0.40,
        frame_threshold: int = 15,
        window_size: int = 8,
        ear_threshold: float = 0.23,
        ear_weight: float = 0.10,
    ):
        self.closed_on_threshold = closed_on_threshold
        self.open_off_threshold = open_off_threshold
        self.frame_threshold = frame_threshold
        self.window = deque(maxlen=window_size)
        self.ear_threshold = ear_threshold
        self.ear_weight = ear_weight
        self.counter = 0

    def reset(self):
        self.window.clear()
        self.counter = 0

    def update(self, left_result, right_result, ear=None):
        if left_result is None or right_result is None:
            self.reset()
            return {
                "is_drowsy": False,
                "score": 0.0,
                "counter": self.counter,
                "reason": "missing_eye_prediction",
            }

        left_closed = left_result["closed_prob"]
        right_closed = right_result["closed_prob"]

        cnn_score = (left_closed + right_closed) / 2.0

        if ear is not None:
            # If EAR is below threshold, this becomes a weak extra closed-eye signal.
            ear_closed_score = np.clip((self.ear_threshold - ear) / self.ear_threshold, 0.0, 1.0)
            score = (1.0 - self.ear_weight) * cnn_score + self.ear_weight * ear_closed_score
        else:
            score = cnn_score

        self.window.append(score)
        smoothed_score = float(np.mean(self.window))

        if smoothed_score >= self.closed_on_threshold:
            self.counter += 1
        elif smoothed_score <= self.open_off_threshold:
            self.counter = 0
        # If in the middle band, keep the counter as-is.

        is_drowsy = self.counter >= self.frame_threshold

        return {
            "is_drowsy": is_drowsy,
            "score": smoothed_score,
            "counter": self.counter,
            "reason": "ok",
        }