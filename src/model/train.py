from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup
import torchaudio
from model.model import PretrainedGenreTransformer, evaluate_model
from config.config import SAMPLE_RATE, TABLE, TRANSFORMER_MODEL_NAME, NUM_LABELS
import numpy as np
from torch.amp import autocast, GradScaler

from model.EarlyStopping import EarlyStopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenreDataset(Dataset):
    def __init__(self, metadata_df, audio_dir):
        self.df = metadata_df
        self.audio_dir = Path(audio_dir)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            TRANSFORMER_MODEL_NAME
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_path = self.audio_dir / f"{row['yt_id']}_chunk_{row['chunk_index']}.wav"
        waveform, sr = torchaudio.load(file_path)

        raw_label = row["genres"]
        if isinstance(raw_label, (list, tuple, np.ndarray, pd.Series)):
            raw_label = raw_label[0]

        label_id = TABLE[raw_label] if isinstance(raw_label, str) else int(raw_label)
        label = torch.tensor(label_id, dtype=torch.long)

        inputs = self.feature_extractor(
            waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )

        return inputs["input_values"].squeeze(0), label


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_with_seed(seed: int = None, verbose: bool = True):
    if seed is None:
        seed = np.random.randint(0, 10000)

    train_df = pd.read_json("dataset/p4_dataset_train.json")
    val_df = pd.read_json("dataset/p4_dataset_val.json")

    audio_dir = "dataset/audio_chunks"
    train_dataset = GenreDataset(train_df, audio_dir)
    val_dataset = GenreDataset(val_df, audio_dir)

    g = torch.Generator()
    g.manual_seed(seed)

    BATCH_SIZE = 20
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        generator=g,
    )

    n_classes = NUM_LABELS

    model = PretrainedGenreTransformer(n_classes).to(DEVICE)

    EPOCHS = 30

    optimizer = torch.optim.AdamW(
        [
            {"params": model.ast.parameters(), "lr": 1e-6},
            {"params": model.classifier.parameters(), "lr": 1e-5},
        ],
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps
    )

    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    best_state = None

    # for name, param in model.ast.named_parameters():
    #     if not any(
    #         layer in name
    #         for layer in [
    #             "encoder.layer.10",
    #             "encoder.layer.11",
    #         ]
    #     ):
    #         param.requires_grad = False

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for inputs, labels in progress_bar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast("cuda"):
                logits = model(inputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_loss += loss.detach()

        train_loss = total_loss.item() / len(train_loader)
        val_metrics = evaluate_model(model, val_loader)

        if verbose:
            print(
                f"Epoch {epoch:04d} | train_loss {train_loss:.4f} | val_loss {val_metrics['loss']:.4f} | val_f1_micro {val_metrics['f1_micro']:.4f} | val_f1_macro {val_metrics['f1_macro']:.4f} | val_acc {val_metrics['accuracy']:.4f}"
            )

        if val_metrics["f1_macro"] > early_stopping.best_score:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if early_stopping(val_metrics["f1_macro"]):
            if verbose:
                print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_model(model, val_loader)
    if verbose:
        print("Final validation metrics:", final_metrics)

    return final_metrics["f1_macro"], model


def main() -> None:
    SEEDS = [42]
    f1_macros = []
    best_model = None
    for seed in SEEDS:
        print(f"Running training with seed {seed}...")
        f1_macro, model = run_with_seed(seed=seed, verbose=True)
        if f1_macro > max(f1_macros, default=0):
            best_model = model
        f1_macros.append(f1_macro)
        print(f"Seed {seed} | F1 Macro: {f1_macro:.4f}")

    torch.save(
        {
            "state_dict": best_model.state_dict(),
        },
        "model/final_model.pth",
    )

    avg_f1_macro = sum(f1_macros) / len(f1_macros)
    print(f"Average F1 Macro over seeds: {avg_f1_macro:.4f}")
    print(f"F1 Macro Std Dev: {np.std(f1_macros):.4f}")
    print(f"F1 Macro Min: {min(f1_macros):.4f} | F1 Macro Max: {max(f1_macros):.4f}")
