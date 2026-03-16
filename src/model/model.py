from typing import Dict, List

import torch.nn as nn
from transformers import ASTModel
from sklearn.metrics import f1_score

from config.config import TRANSFORMER_MODEL_NAME, NUM_LABELS
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PretrainedGenreTransformer(nn.Module):
    def __init__(self, n_classes, dropout_prob=0.2):
        super().__init__()

        self.ast = ASTModel.from_pretrained(TRANSFORMER_MODEL_NAME)

        hidden_size = self.ast.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, input_values):
        outputs = self.ast(input_values=input_values)
        pooled = outputs.last_hidden_state[:, 0]

        logits = self.classifier(pooled)

        return logits


def _compute_f1(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "f1_micro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "f1_per_class": {},
        }

    labels = list(range(NUM_LABELS))
    f1_micro = f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0,
    )
    f1_per_class_scores = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": {
            label: float(score) for label, score in zip(labels, f1_per_class_scores)
        },
    }


def evaluate_model(
    model: nn.Module,
    val_loader,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()

    model.eval()

    loss_sum = 0.0
    total_samples = 0
    correct_preds = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)

            loss_sum += criterion(logits, labels).item()

            predictions = torch.argmax(logits, dim=1)

            correct_preds += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            y_pred.extend(predictions.detach().cpu().tolist())
            y_true.extend(labels.detach().cpu().tolist())

    f1_scores = _compute_f1(y_true, y_pred)
    average_loss = loss_sum / len(val_loader) if len(val_loader) else 0.0
    accuracy = (correct_preds / total_samples) if total_samples > 0 else 0.0

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        **f1_scores,
    }
