import cv2
import torch
import numpy as np


def preprocess_eye(eye_img, size=224):
    """
    Input:
        eye_img : BGR eye image (NumPy array)
    Output:
        torch tensor of shape (1, 3, 224, 224)
    """

    # 1. Convert BGR → RGB
    rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)

    # 2. Resize to MobileNet input size
    resized = cv2.resize(rgb, (size, size))

    # 3. Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # 4. Convert to torch tensor (H, W, C) → (C, H, W)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)

    # 5. ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    # 6. Add batch dimension → (1, 3, 224, 224)
    tensor = tensor.unsqueeze(0)

    return tensor


#converted into rgb model
