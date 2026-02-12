# src/drowsiness_logic.py

class DrowsinessDetector:
    def __init__(self, frame_threshold=30):
        """
        frame_threshold:
            Number of consecutive CLOSED-eye frames
            required to declare drowsiness
        """
        self.frame_threshold = frame_threshold
        self.closed_counter = 0
        self.drowsy = False

    def update(self, left_closed, right_closed):
        """
        Update drowsiness state for the current frame.

        Inputs:
            left_closed  : True / False / None
            right_closed : True / False / None

        Returns:
            drowsy (bool)
        """

        # If eyes are not reliably detected, reset
        if left_closed is None or right_closed is None:
            self.closed_counter = 0
            self.drowsy = False
            return self.drowsy

        # Both eyes closed → increment counter
        if left_closed and right_closed:
            self.closed_counter += 1
        else:
            # Any eye open → reset
            self.closed_counter = 0
            self.drowsy = False

        # Check threshold
        if self.closed_counter >= self.frame_threshold:
            self.drowsy = True

        return self.drowsy
