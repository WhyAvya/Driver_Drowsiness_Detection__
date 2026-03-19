import torch
import torch.nn.functional as F
from models.model_arch import EyeStateCNN
import os


class EyeStateInference:

    def __init__(self, model_path="models/eye_state_mobilenet.pth", device=None):
        """
        model_path: path to .pth file
        device: 'cpu' or 'cuda'
        """

        # -------------------------------
        # Device setup
        # -------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print("Running on device:", self.device)

        # -------------------------------
        # Load model
        # -------------------------------
        self.model = EyeStateCNN().to(self.device)

        if os.path.exists(model_path):

            state_dict = torch.load(
                model_path,
                map_location=self.device
            )

            self.model.load_state_dict(state_dict)

            print("✅ Loaded trained CNN weights")

        else:
            print("⚠️ Model weights not found → using random weights")

        self.model.eval()

        # -------------------------------
        # Confidence thresholds
        # -------------------------------
        self.min_confidence = 0.65      # ignore weak predictions
        self.closed_strict = 0.75       # stricter for CLOSED


    # ============================================================
    # Predict eye state
    # ============================================================
    def predict(self, eye_tensor):
        """
        Input:
            eye_tensor: (1,3,224,224)

        Returns:
            label:
                0 → closed
                1 → open
                None → uncertain

            confidence: float
        """

        if eye_tensor is None:
            return None, 0.0

        with torch.no_grad():

            eye_tensor = eye_tensor.to(self.device).float()

            # Forward pass
            logits = self.model(eye_tensor)

            probs = F.softmax(logits, dim=1)

            confidence, label = torch.max(probs, dim=1)

            confidence = confidence.item()
            label = label.item()

        # -------------------------------
        # Confidence filtering
        # -------------------------------

        # Ignore weak predictions
        if confidence < self.min_confidence:
            return None, confidence

        # Be stricter for CLOSED (reduces false alarms)
        if label == 0 and confidence < self.closed_strict:
            return None, confidence

        return label, confidence