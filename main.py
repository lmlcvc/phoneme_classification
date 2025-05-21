# --------------------------- IMPORTS --------------------------- #
import os
import numpy as np
import librosa
from textgrid import TextGrid
import torch
import logging
import argparse

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scipy.spatial.distance import mahalanobis
from joblib import Parallel, delayed
from tqdm import tqdm
import configparser

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from mahalanobis_net import MahalanobisNet, compute_class_stats, mahalanobis_predict


# ----------------------------- CONFIG ----------------------------- #
config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

male_audios_dir = config['male_audios_dir']
female_audios_dir = config['female_audios_dir']
textgrids_dir = config['textgrids_dir']

LPC_ORDER = 12
VAL_RATIO = 0.2
FRAME_LENGTH = 0.025
FRAME_STEP = 0.010
REGULARIZATION = 1e-6
DROP_LABELS = {'[]', 'greska'}

logging.basicConfig(filename='execution_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Starting execution...")


# ------------------------ PHONEME ALIGNMENT ------------------------ #
def align_phonemes_to_frames(wav_path, grid_path, silence_threshold=0.3, keep_unknown=False):
    y, sr = librosa.load(wav_path, sr=None)
    frames = librosa.util.frame(
        y,
        frame_length=int(FRAME_LENGTH * sr),
        hop_length=int(FRAME_STEP * sr)
    ).T
    tg = TextGrid.fromFile(grid_path)
    phoneme_tier = next(t for t in tg.tiers if t.name.lower() in ["phones", "phoneme", "phonemes"])
    frame_times = librosa.frames_to_time(np.arange(frames.shape[0]), sr=sr, hop_length=int(FRAME_STEP * sr))

    frame_labels = []
    for t in frame_times:
        label = None
        for interval in phoneme_tier.intervals:
            if interval.minTime <= t < interval.maxTime:
                raw_mark = interval.mark.strip()
                if raw_mark == "":
                    duration = interval.maxTime - interval.minTime
                    label = 'sil' if duration <= silence_threshold else None
                elif raw_mark.lower() in ['spn', 'unk', '???']:
                    label = '<unk>' if keep_unknown else None
                else:
                    label = raw_mark
                break
        frame_labels.append(label)

    filtered_frames = []
    filtered_labels = []
    for f, l in zip(frames, frame_labels):
        if l is not None:
            filtered_frames.append(f)
            filtered_labels.append(l)

    return np.array(filtered_frames), filtered_labels


# --------------------------- DATA LOADER --------------------------- #
def load_aligned_data(audio_dir, textgrid_dir):
    data = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue
            audio_path = os.path.join(root, file)
            speaker_id = os.path.basename(root)
            textgrid_path = os.path.join(textgrid_dir, speaker_id, file.replace('.wav', '.TextGrid'))
            if not os.path.exists(textgrid_path):
                continue
            try:
                frames, labels = align_phonemes_to_frames(audio_path, textgrid_path)
                if frames.shape[0] != len(labels):
                    continue
                data.append((frames, labels))
            except Exception as e:
                continue
    return data


# --------------------------- LPC FEATURES --------------------------- #
def extract_lpc_features(frames):
    lpc_list = []
    for frame in frames:
        if len(frame) <= LPC_ORDER:
            continue
        try:
            coeffs = librosa.lpc(frame, order=LPC_ORDER)
            lpc_list.append(coeffs)
        except Exception:
            continue
    return np.array(lpc_list)


# ------------------------- MAHALANOBIS CLF -------------------------- #
def mahalanobis_classification(X, means, inv_covs):
    preds = []
    for x in tqdm(X, desc="Classifying"):
        dists = [mahalanobis(x, means[label], inv_covs[label]) for label in means]
        preds.append(np.argmin(dists))
    return np.array(preds)


# ----------------------- CLASSIFIER COMPARISON ----------------------- #
def compare_mahalanobis_classifiers(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, classifiers_to_run):
    print(f"\n\nRunning selected classifiers: {', '.join(classifiers_to_run)}\n")

    classes = np.unique(y_train)
    cov = np.cov(X_train.T) + REGULARIZATION * np.eye(X_train.shape[1])
    inv_cov = np.linalg.inv(cov)
    means = {label: np.mean(X_train[y_train == label], axis=0) for label in classes}

    if "centroid" in classifiers_to_run:
        print("[Running] Centroid Mahalanobis Classifier...")
        def centroid_predict(X):
            preds = [classes[np.argmin([mahalanobis(x, means[c], inv_cov) for c in classes])] for x in X]
            return np.array(preds)

        y_pred = centroid_predict(X_test)
        print("Centroid Mahalanobis Accuracy:", accuracy_score(y_test, y_pred) * 100)
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    if "lda" in classifiers_to_run:
        print("\n[Running] Linear Discriminant Analysis (LDA)...")
        try:
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            y_pred_lda = clf.predict(X_test)
            print("LDA Accuracy:", accuracy_score(y_test, y_pred_lda) * 100)
            print(classification_report(y_test, y_pred_lda, target_names=label_encoder.classes_))
        except Exception as e:
            print(f"LDA failed: {e}")

    if "mahalanobisnet" in classifiers_to_run:
        print("\n[Running] MahalanobisNet Classifier...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MahalanobisNet(input_dim=X_train.shape[1], embedding_dim=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=512, shuffle=True)
        for epoch in range(10):
            model.train()
            for xb, yb in loader:
                embeds = model(xb)
                means, inv_cov = compute_class_stats(embeds, yb, len(label_encoder.classes_))
                preds = mahalanobis_predict(embeds, means, inv_cov)
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_embeds = model(X_val_tensor)
                train_embeds = model(X_train_tensor)
                means, inv_cov = compute_class_stats(train_embeds, y_train_tensor, len(label_encoder.classes_))
                val_preds = mahalanobis_predict(val_embeds, means, inv_cov)
                val_pred_labels = val_preds.argmax(dim=1)
                val_acc = (val_pred_labels == y_val_tensor).float().mean().item()
                print(f"Epoch {epoch+1}: Val Accuracy = {val_acc * 100:.2f}%")

        with torch.no_grad():
            test_embeds = model(X_test_tensor)
            test_preds = mahalanobis_predict(test_embeds, means, inv_cov)
            test_pred_labels = test_preds.argmax(dim=1)
            test_acc = (test_pred_labels == y_test_tensor).float().mean().item()
            print(f"\n[MahalanobisNet] Test Accuracy: {test_acc * 100:.2f}%")
            print(classification_report(
                y_test_tensor.cpu().numpy(),
                test_pred_labels.cpu().numpy(),
                target_names=label_encoder.classes_
            ))


# ------------------------------ MAIN ------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Mahalanobis-based classifiers.")
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        default=["centroid", "lda", "mahalanobisnet"],
        help="List of classifiers to include. Options: centroid, lda, mahalanobisnet"
    )
    parser.add_argument(
        "--run-minimal-dataset",
        action="store_true",
        help="Run on a minimal dataset."
    )
    args = parser.parse_args()

    data_m = load_aligned_data(male_audios_dir, textgrids_dir)
    data_f = load_aligned_data(female_audios_dir, textgrids_dir)
    all_data = data_m + data_f 
    if args.run_minimal_dataset:
        all_data = all_data[:100]

    frame_data = [frames for frames, _ in all_data]
    frame_labels = [labels for _, labels in all_data]
    lpc_features = [extract_lpc_features(frames) for frames in tqdm(frame_data, desc="LPC Extraction")]

    filtered_features = []
    filtered_labels = []
    for features, labels in zip(lpc_features, frame_labels):
        if features.shape[0] > 0:
            filtered_features.append(features)
            filtered_labels.append(labels)

    all_features = np.vstack(filtered_features)
    all_labels = np.array([lbl for seq in filtered_labels for lbl in seq])

    mask = np.array([label not in DROP_LABELS for label in all_labels])
    all_features = all_features[mask]
    all_labels = all_labels[mask]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(all_features, y_encoded, test_size=0.2, random_state=42)
    val_size = int(X_train.shape[0] * VAL_RATIO)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    compare_mahalanobis_classifiers(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, args.classifiers)

