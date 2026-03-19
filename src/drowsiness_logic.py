class DrowsinessDetector:

    def __init__(self, frame_threshold=30):

        self.frame_threshold = frame_threshold
        self.closed_counter = 0
        self.drowsy = False


    def update(self, left_closed, right_closed):
        """
        Inputs:
            left_closed  : True / False / None
            right_closed : True / False / None

        Returns:
            drowsy (bool)
        """

        # -------------------------------
        # Case 1: Both eyes missing → ignore frame
        # -------------------------------
        if left_closed is None and right_closed is None:
            return self.drowsy

        # -------------------------------
        # Case 2: Determine closed state
        # -------------------------------

        closed_score = 0

        if left_closed is True:
            closed_score += 1

        if right_closed is True:
            closed_score += 1

        # -------------------------------
        # Case 3: Update counter
        # -------------------------------

        # Both eyes closed → strong signal
        if closed_score == 2:
            self.closed_counter += 1

        # One eye closed → weak signal (still count)
        elif closed_score == 1:
            self.closed_counter += 0.5

        # Eyes open → reset
        else:
            self.closed_counter = 0
            self.drowsy = False

        # -------------------------------
        # Case 4: Check threshold
        # -------------------------------
        if self.closed_counter >= self.frame_threshold:
            self.drowsy = True

        return self.drowsy