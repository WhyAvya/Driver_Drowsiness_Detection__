import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from models.model_arch import EyeStateEfficientNet
from src.preprocess import get_train_transforms, get_eval_transforms


# =========================
# CONFIG
# =========================
TRAIN_DIR = "data/dataset/train"
VAL_DIR = "data/dataset/val"
TEST_DIR = "data/dataset/test"

MODEL_SAVE_PATH = "models/best_eye_state_effnet.pth"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# DATA
# =========================
train_dataset = ImageFolder(TRAIN_DIR, transform=get_train_transforms())
val_dataset = ImageFolder(VAL_DIR, transform=get_eval_transforms())
test_dataset = ImageFolder(TEST_DIR, transform=get_eval_transforms())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Classes:", train_dataset.classes)
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))


# =========================
# MODEL
# =========================
model = EyeStateEfficientNet(num_classes=2, pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


# =========================
# TRAIN / VALIDATE FUNCTION
# =========================
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds


# =========================
# TRAINING LOOP
# =========================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_acc, _, _ = run_epoch(val_loader, train=False)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("models", exist_ok=True)

        checkpoint = {
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "class_to_idx": train_dataset.class_to_idx,
            "idx_to_class": {v: k for k, v in train_dataset.class_to_idx.items()}
        }

        torch.save(checkpoint, MODEL_SAVE_PATH)
        print("✅ Best model saved!")


# =========================
# TEST BEST MODEL
# =========================
print("\nLoading best model for test evaluation...")

checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_loss, test_acc, test_labels, test_preds = run_epoch(test_loader, train=False)

print("\n=== FINAL TEST RESULTS ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc : {test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))

print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_preds))