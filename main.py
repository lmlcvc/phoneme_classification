# --------------------------- IMPORTS --------------------------- #
import os
import numpy as np
import librosa
from textgrid import TextGrid
import torch
import logging
import argparse
import datetime
import configparser

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import mahalanobis
from joblib import Parallel, delayed
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from mahalanobis_net import MahalanobisNet, MahalanobisRNN, compute_class_stats, mahalanobis_scores

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


# ----------------------- SEQUENCE GENERATION ----------------------- #
def make_sequences(features_list, labels_list, seq_len=15):
    X, y = [], []
    for features, labels in zip(features_list, labels_list):
        if len(features) < seq_len:
            continue
        for i in range(len(features) - seq_len + 1):
            label_seq = labels[i:i+seq_len]
            if any(lbl in DROP_LABELS for lbl in label_seq):
                continue
            X.append(features[i:i+seq_len])
            y.append(label_seq[seq_len // 2])  # label from center frame
    return np.array(X), np.array(y)


# ----------------------- CLASSIFIER COMPARISON ----------------------- #
def compare_mahalanobis_classifiers(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, classifiers_to_run):
    print(f"\n\nRunning selected classifiers: {', '.join(classifiers_to_run)}\n")
    classes = np.unique(y_train)

    if "centroid" in classifiers_to_run:
        print("[Running] Centroid Mahalanobis Classifier...")
        cov = np.cov(X_train.T) + REGULARIZATION * np.eye(X_train.shape[1])
        inv_cov = np.linalg.inv(cov)
        means = {label: np.mean(X_train[y_train == label], axis=0) for label in classes}
        preds = [min(means, key=lambda c: mahalanobis(x, means[c], inv_cov)) for x in X_test]
        print("Centroid Accuracy:", accuracy_score(y_test, preds) * 100)

    if "lda" in classifiers_to_run:
        print("\n[Running] Linear Discriminant Analysis (LDA)...")
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("LDA Accuracy:", accuracy_score(y_test, y_pred) * 100)

    if "mahalanobisnet" in classifiers_to_run:
        print("\n[Running] MahalanobisNet Classifier...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = MahalanobisNet(input_dim=X_train.shape[1], embedding_dim=32, n_classes=len(label_encoder.classes_)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=512, shuffle=True)

        for epoch in range(10):
            model.train()
            for xb, yb in train_loader:
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation using Mahalanobis distance on embeddings
            model.eval()
            with torch.no_grad():
                _, train_embeds = model(X_train_tensor)
                _, val_embeds = model(X_val_tensor)

                means, inv_cov = compute_class_stats(train_embeds, y_train_tensor, len(label_encoder.classes_))
                val_scores = mahalanobis_scores(val_embeds, means, inv_cov)
                val_pred_labels = val_scores.argmax(dim=1)
                val_acc = (val_pred_labels == y_val_tensor).float().mean().item()

                print(f"Epoch {epoch + 1}: Validation Accuracy (MahalanobisNet) = {val_acc * 100:.2f}%")

        # Final test evaluation
        model.eval()
        with torch.no_grad():
            _, train_embeds = model(X_train_tensor)
            _, test_embeds = model(X_test_tensor)

            means, inv_cov = compute_class_stats(train_embeds, y_train_tensor, len(label_encoder.classes_))
            test_scores = mahalanobis_scores(test_embeds, means, inv_cov)
            test_pred_labels = test_scores.argmax(dim=1)
            test_acc = (test_pred_labels == y_test_tensor).float().mean().item()

            print(f"\n[MahalanobisNet] Test Accuracy: {test_acc * 100:.2f}%")
            print(classification_report(
                y_test_tensor.cpu().numpy(),
                test_pred_labels.cpu().numpy(),
                target_names=label_encoder.classes_
            ))

    if "rnn" in classifiers_to_run:
        print("\n[Running] RNN Classifier...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MahalanobisRNN(input_dim=X_train.shape[2], hidden_dim=64, n_classes=len(label_encoder.classes_)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        for epoch in range(10):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate on validation set using Mahalanobis distance on embeddings
            model.eval()
            with torch.no_grad():
                _, train_embeds = model(X_train_tensor)
                means, inv_cov = compute_class_stats(train_embeds, y_train_tensor, len(label_encoder.classes_))

                _, val_embeds = model(X_val_tensor)
                val_scores = mahalanobis_scores(val_embeds, means, inv_cov)
                val_pred_labels = val_scores.argmax(dim=1)
                val_acc = (val_pred_labels == y_val_tensor).float().mean().item()
                print(f"Epoch {epoch + 1}: Val Accuracy (Mahalanobis) = {val_acc * 100:.2f}%")

        # Final test evaluation using Mahalanobis distance
        model.eval()
        with torch.no_grad():
            _, train_embeds = model(X_train_tensor)
            means, inv_cov = compute_class_stats(train_embeds, y_train_tensor, len(label_encoder.classes_))

            _, test_embeds = model(X_test_tensor)
            test_scores = mahalanobis_scores(test_embeds, means, inv_cov)
            test_pred_labels = test_scores.argmax(dim=1)
            test_acc = (test_pred_labels == y_test_tensor).float().mean().item()
            print(f"\n[RNN + Mahalanobis] Test Accuracy: {test_acc * 100:.2f}%")

# ------------------------------ MAIN ------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("classifiers", nargs="+", choices=["centroid", "lda", "mahalanobisnet", "rnn"], help="Classifier(s) to run")
    parser.add_argument("--run-minimal-dataset", action="store_true")
    args = parser.parse_args()

    print("Loading data...")
    data = load_aligned_data(male_audios_dir, textgrids_dir) + load_aligned_data(female_audios_dir, textgrids_dir)
    if args.run_minimal_dataset:
        data = data[:100]

    features = [extract_lpc_features(f) for f, _ in data]
    labels = [l for _, l in data]

    filtered_features, filtered_labels = [], []
    for f, l in zip(features, labels):
        if f.shape[0] > 0:
            filtered_features.append(f)
            filtered_labels.append(l)

    all_features = np.vstack(filtered_features)
    all_labels = np.array([l for seq in filtered_labels for l in seq])

    mask = np.array([l not in DROP_LABELS for l in all_labels])
    all_features = all_features[mask]
    all_labels = all_labels[mask]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(all_labels)

    if "rnn" in args.classifiers:
        X_seq, y_seq_raw = make_sequences(filtered_features, filtered_labels)
        y_seq = label_encoder.transform(y_seq_raw)  
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(all_features, y_encoded, test_size=0.2, random_state=42)

    val_size = int(X_train.shape[0] * VAL_RATIO)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    if "rnn" not in args.classifiers:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    compare_mahalanobis_classifiers(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, args.classifiers)
