import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from models.model_arch import EyeStateCNN
from tqdm import tqdm


# ================= CONFIG =================
DATASET_DIR = "data/dataset"   # <-- CHANGE this if your folder name is different
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 4
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "eye_state_mobilenet.pth"


# ================= DATASET =================
class EyeDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir should be:
        - data/dataset/train
        - data/dataset/test
        """
        self.samples = []

        class_map = {
            "sleepy": 0,  # CLOSED
            "awake": 1    # OPEN
        }

        for class_name, label in class_map.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img in os.listdir(class_dir):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, img)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = torch.tensor(img).permute(2, 0, 1)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        return img, torch.tensor(label, dtype=torch.long)


# ================= TRAIN =================
def train():
    print(f"Using device: {DEVICE}")

    train_dataset = EyeDataset(os.path.join(DATASET_DIR, "train"))
    test_dataset = EyeDataset(os.path.join(DATASET_DIR, "test"))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = EyeStateCNN(freeze_backbone=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    for epoch in range(EPOCHS):
        # -------- Training --------
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        print(f"Epoch {epoch+1} | Train Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # -------- Validation --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total * 100
        print(f"Epoch {epoch+1} | Validation Acc: {val_acc:.2f}%\n")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved as {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()