import numpy as np
import torch

from models.model_arch import EyeStateEfficientNet
from src.preprocess import preprocess_eye


class EyeStateInference:
    def __init__(self, model_path: str = "models/best_eye_state_effnet.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        # If checkpoint is a dict with metadata, load it properly.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.class_to_idx = checkpoint.get("class_to_idx", {"closed": 0, "open": 1})
            self.idx_to_class = checkpoint.get("idx_to_class", {0: "closed", 1: "open"})
            model_state = checkpoint["model_state_dict"]
        else:
            self.class_to_idx = {"closed": 0, "open": 1}
            self.idx_to_class = {0: "closed", 1: "open"}
            model_state = checkpoint

        self.closed_index = self.class_to_idx.get("closed", 0)
        self.open_index = self.class_to_idx.get("open", 1)

        self.model = EyeStateEfficientNet(num_classes=2, pretrained=False).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, eye_crop):
        tensor = preprocess_eye(eye_crop)
        if tensor is None:
            return None

        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = self.idx_to_class.get(pred_idx, str(pred_idx))

        return {
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "confidence": float(probs[pred_idx]),
            "closed_prob": float(probs[self.closed_index]),
            "open_prob": float(probs[self.open_index]),
            "probs": probs.tolist(),
        }