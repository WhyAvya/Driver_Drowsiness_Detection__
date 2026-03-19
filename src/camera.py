import cv2


class Camera:

    def __init__(self, index=0, width=640, height=480, fps=30):

        # Enable OpenCV optimizations
        cv2.setUseOptimized(True)

        # Open camera using DirectShow (low latency on Windows)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("Webcam not accessible")

        # Reduce resolution for faster processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Limit FPS
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Reduce internal buffering (very important for lag)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError("Failed to grab frame")

        return frame

    def release(self):

        if self.cap.isOpened():
            self.cap.release()