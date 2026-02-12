import torch
import torch.nn.functional as F
from models.model_arch import EyeStateCNN
import os


class EyeStateInference:
    def __init__(self, model_path=None, device="cpu"):
        """
        model_path: path to .pth file
        device: 'cpu' or 'cuda'
        """
        self.device = device

        # Load model architecture (MobileNetV2 based)
        self.model = EyeStateCNN().to(self.device)

        # Load trained weights if provided
        if model_path is not None and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print("✅ Loaded trained CNN weights")
        else:
            print("⚠️ Using untrained CNN weights (for system testing)")

        # Set model to inference mode
        self.model.eval()

    def predict(self, eye_tensor):
        """
        eye_tensor: torch.Tensor of shape (1, 3, 224, 224)

        Returns:
            label: 0 (closed) or 1 (open)
            confidence: probability score
        """

        with torch.no_grad():
            # Ensure correct dtype and device
            eye_tensor = eye_tensor.to(self.device).float()

            # Forward pass (logits)
            logits = self.model(eye_tensor)

            # Convert logits → probabilities
            probs = F.softmax(logits, dim=1)

            # Get predicted class
            confidence, label = torch.max(probs, dim=1)

        return label.item(), confidence.item()