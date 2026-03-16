import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from config.config import AUDIO_LENGTH, SAMPLE_RATE
import os


def load_as_waveform(file_path):
    waveform, sr = torchaudio.load(file_path)
    return waveform, sr


def convert_to_mono(waveform):
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def resample_waveform(waveform, sr):
    target_sr = SAMPLE_RATE
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        return resampler(waveform), target_sr
    return waveform, sr


def create_chunks(waveform):
    sr = SAMPLE_RATE
    chunk_length = AUDIO_LENGTH
    chunk_size = int(sr * chunk_length)
    total_length = waveform.shape[0]

    skip_ratio = 0.15
    start_idx = int(total_length * skip_ratio)
    end_idx = int(total_length * (1 - skip_ratio))
    if end_idx <= start_idx:
        trimmed = waveform[start_idx:end_idx]
        padding = chunk_size - trimmed.shape[0]
        return [torch.nn.functional.pad(trimmed, (0, padding))]

    trimmed = waveform[start_idx:end_idx]
    trimmed_length = trimmed.shape[0]

    if trimmed_length <= chunk_size:
        padding = chunk_size - trimmed_length
        return [torch.nn.functional.pad(trimmed, (0, padding))]

    chunks = []
    for start in range(0, trimmed_length, chunk_size):
        end = min(start + chunk_size, trimmed_length)
        chunk = trimmed[start:end]
        if chunk.shape[0] < chunk_size:
            padding = chunk_size - chunk.shape[0]
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        chunks.append(chunk)

    return chunks


def sqeeze_audiowaveform(waveform):
    if waveform.shape[0] == 1:
        return waveform.squeeze(0)
    return waveform


def remove_chunks_with_silence(chunks, threshold=0.1):
    non_silent_chunks = []
    for chunk in chunks:
        if chunk.abs().mean() > threshold:
            non_silent_chunks.append(chunk)
    return non_silent_chunks


def preprocess_audio(file_path):
    waveform, sr = load_as_waveform(file_path)
    waveform = convert_to_mono(waveform)
    waveform, sr = resample_waveform(waveform, sr)
    waveform = sqeeze_audiowaveform(waveform)
    chunks = create_chunks(waveform)
    chunks = remove_chunks_with_silence(chunks)
    return chunks


def save_chunks(chunks, base_path, yt_id):
    os.makedirs(base_path, exist_ok=True)
    for i, chunk in enumerate(chunks):
        chunk_path = f"{base_path}/{yt_id}_chunk_{i}.wav"
        torchaudio.save(chunk_path, chunk, SAMPLE_RATE)


def augment_df(df):
    augmented_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
        file_path = row["audio_path"]
        chunks = preprocess_audio(file_path)
        for i, chunk in enumerate(chunks):
            new_row = row.copy()
            new_row["chunk_index"] = i
            augmented_rows.append(new_row)
        save_chunks(chunks, "dataset/audio_chunks", row["yt_id"])
    return pd.DataFrame(augmented_rows)


def preprocess_dataset():
    if os.path.exists("dataset/audio_chunks"):
        for file in os.listdir("dataset/audio_chunks"):
            file_path = os.path.join("dataset/audio_chunks", file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    train_df = pd.read_json("dataset/p3_dataset_train.json")

    train_df = augment_df(train_df)
    train_df = train_df.reset_index(drop=True)

    val_df = pd.read_json("dataset/p3_dataset_val.json")
    val_df = augment_df(val_df)
    val_df = val_df.reset_index(drop=True)

    train_df.to_json("dataset/p4_dataset_train.json", index=False)
    val_df.to_json("dataset/p4_dataset_val.json", index=False)
