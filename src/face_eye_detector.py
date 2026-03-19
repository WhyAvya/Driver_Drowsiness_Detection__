import cv2
import os

# -------------------------------
# Resolve base project directory
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# -------------------------------
# Haar cascade paths
# -------------------------------
FACE_CASCADE_PATH = os.path.join(
    BASE_DIR, "cascades", "haarcascade_frontalface_default.xml"
)

EYE_CASCADE_PATH = os.path.join(
    BASE_DIR, "cascades", "haarcascade_eye_tree_eyeglasses.xml"
)

# -------------------------------
# Load cascades ONCE
# -------------------------------
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError("Face cascade failed to load")

if eye_cascade.empty():
    raise RuntimeError("Eye cascade failed to load")


# ============================================================
# Face + Eye Detection (Improved)
# ============================================================
def detect_face_and_eyes(frame, draw=False):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # -------------------------------
    # Detect faces
    # -------------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None, None, None, frame

    # Pick largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # -------------------------------
    # Use upper 60% of face (IMPORTANT)
    # -------------------------------
    roi_h = int(h * 0.6)

    face_gray = gray[y:y + roi_h, x:x + w]
    face_color = frame[y:y + roi_h, x:x + w]

    # -------------------------------
    # Detect eyes
    # -------------------------------
    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    print("Eyes detected:", len(eyes))

    # Sort left → right
    eyes = sorted(eyes, key=lambda e: e[0])

    left_eye = None
    right_eye = None

    pad = 6  # 👈 important for closed eyes

    # -------------------------------
    # Extract LEFT eye (if exists)
    # -------------------------------
    if len(eyes) >= 1:
        ex, ey, ew, eh = eyes[0]

        x1 = max(0, ex - pad)
        y1 = max(0, ey - pad)
        x2 = min(face_color.shape[1], ex + ew + pad)
        y2 = min(face_color.shape[0], ey + eh + pad)

        left_eye = face_color[y1:y2, x1:x2]

    # -------------------------------
    # Extract RIGHT eye (if exists)
    # -------------------------------
    if len(eyes) >= 2:
        ex, ey, ew, eh = eyes[1]

        x1 = max(0, ex - pad)
        y1 = max(0, ey - pad)
        x2 = min(face_color.shape[1], ex + ew + pad)
        y2 = min(face_color.shape[0], ey + eh + pad)

        right_eye = face_color[y1:y2, x1:x2]

    # -------------------------------
    # Draw debug boxes
    # -------------------------------
    if draw:

        # Face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Eye boxes
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(
                frame,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (255, 0, 0),
                2
            )

    return (x, y, w, h), left_eye, right_eye, frame