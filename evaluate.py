import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.model_arch import EyeStateEfficientNet
from src.preprocess import get_eval_transforms

# =========================
# CONFIG
# =========================
TEST_DIR = "data/dataset/test"
MODEL_PATH = "models/best_eye_state_effnet.pth"
BATCH_SIZE = 32

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD TEST DATA
# =========================
test_dataset = ImageFolder(TEST_DIR, transform=get_eval_transforms())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Classes:", test_dataset.classes)
print("Test size:", len(test_dataset))

# =========================
# LOAD MODEL
# =========================
model = EyeStateEfficientNet(num_classes=2, pretrained=False).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================
# EVALUATION
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# =========================
# RESULTS
# =========================
acc = accuracy_score(all_labels, all_preds)

print("\n=== FINAL EVALUATION RESULTS ===")
print(f"Test Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))