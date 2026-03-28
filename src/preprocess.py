import cv2
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================
# TRAINING TRANSFORMS
# =========================
def get_train_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# =========================
# RUNTIME PREPROCESSING
# =========================
def preprocess_eye(eye_crop):
    """
    Converts OpenCV BGR eye crop into model-ready tensor
    for real-time inference.
    """
    if eye_crop is None or eye_crop.size == 0:
        return None

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(gray)

    transform = get_eval_transforms()
    tensor = transform(pil_img).unsqueeze(0)

    return tensor