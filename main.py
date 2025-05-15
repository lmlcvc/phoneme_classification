# --------------------------- IMPORTS --------------------------- #
import os
import numpy as np
import librosa
from textgrid import TextGrid
import torch
import logging

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from scipy.spatial.distance import mahalanobis
from joblib import Parallel, delayed
from tqdm import tqdm
import configparser

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from mahalanobis_net import MahalanobisNet, compute_class_stats, mahalanobis_predict


# TODO separate classes/files?


# ----------------------------- CONFIG ----------------------------- #
config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

# Directories
male_audios_dir = config['male_audios_dir']
female_audios_dir = config['female_audios_dir']
textgrids_dir = config['textgrids_dir']

# Constants
LPC_ORDER = 12
N_JOBS = -1
VAL_RATIO = 0.2
FRAME_LENGTH = 0.025
FRAME_STEP = 0.010
REGULARIZATION = 1e-6
DROP_LABELS = {'[]', 'greska'}

# --------------------------- LOGGING ----------------------------- #
logging.basicConfig(filename='execution_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
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

                # empty label (silence)
                if raw_mark == "":
                    duration = interval.maxTime - interval.minTime
                    if duration <= silence_threshold:
                        label = 'sil'  # keep short silence
                    else:
                        label = None     # drop long silence
                # unknowns
                elif raw_mark.lower() in ['spn', 'unk', '???']:
                    label = '<unk>' if keep_unknown else None
                # known labels
                else:
                    label = raw_mark
                break

        frame_labels.append(label)

    # Filter out dropped frames (None labels)
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
                logging.warning(f"Missing TextGrid for {audio_path}")
                continue

            try:
                frames, labels = align_phonemes_to_frames(audio_path, textgrid_path)
                if frames.shape[0] != len(labels):
                    logging.warning(f"Frame-label mismatch: {file}")
                    continue
                data.append((frames, labels))
            except Exception as e:
                logging.error(f"Error loading {file}: {e}")
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
def compare_mahalanobis_classifiers(X_train, y_train, X_test, y_test, label_encoder):
    print("\n\nComparing Mahalanobis-based classifiers...\n")

    # --- Precompute per-class stats ---
    classes = np.unique(y_train)
    means = {}
    inv_cov = None

    # Shared covariance matrix for Mahalanobis k-NN and LDA
    cov = np.cov(X_train.T)
    cov += REGULARIZATION * np.eye(cov.shape[0])
    inv_cov = np.linalg.inv(cov)

    for label in classes:
        means[label] = np.mean(X_train[y_train == label], axis=0)

    # --- Classifier 1: Mahalanobis Centroid ---
    def centroid_predict(X):
        preds = []
        for x in X:
            dists = [mahalanobis(x, means[c], inv_cov) for c in classes]
            preds.append(classes[np.argmin(dists)])
        return np.array(preds)

    y_pred_centroid = centroid_predict(X_test)
    acc = accuracy_score(y_test, y_pred_centroid)
    print(f"Centroid Mahalanobis Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred_centroid, target_names=label_encoder.classes_))

    # --- Classifier 2: Mahalanobis k-NN ---
    def mahalanobis_knn_predict(X_test, k=5):
        predictions = []
        for x in X_test:
            dists = [mahalanobis(x, x_train, inv_cov) for x_train in X_train]
            knn_indices = np.argsort(dists)[:k]
            knn_labels = y_train[knn_indices]
            voted = np.bincount(knn_labels).argmax()
            predictions.append(voted)
        return np.array(predictions)

    for k in [3, 5, 7]:
        y_pred_knn = mahalanobis_knn_predict(X_test, k=k)
        acc = accuracy_score(y_test, y_pred_knn)
        print(f"Mahalanobis k-NN (k={k}) Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred_knn, target_names=label_encoder.classes_))

    # --- Classifier 3: LDA (implicitly Mahalanobis-like) ---
    try:
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_train, y_train)
        y_pred_lda = clf_lda.predict(X_test)
        acc = accuracy_score(y_test, y_pred_lda)
        print(f"LDA Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred_lda, target_names=label_encoder.classes_))
    except Exception as e:
        print(f"LDA failed: {e}")

    # --- Classifier 4: Mahalanobis Neural Classifier ---
    print("\nTraining MahalanobisNet (shallow MLP + Mahalanobis distance)...")

    input_dim = X_train.shape[1]
    n_classes = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MahalanobisNet(input_dim=input_dim, embedding_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    for epoch in range(10):  # You can increase this later
        model.train()
        for xb, yb in train_loader:
            embeds = model(xb)
            means, inv_cov = compute_class_stats(embeds, yb, n_classes)
            preds = mahalanobis_predict(embeds, means, inv_cov)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_embeds = model(X_val_tensor)
            train_embeds = model(X_train_tensor)
            means, inv_cov = compute_class_stats(train_embeds, y_train_tensor, n_classes)
            val_preds = mahalanobis_predict(val_embeds, means, inv_cov)
            val_acc = (val_preds == y_val_tensor).float().mean().item()
            print(f"Epoch {epoch+1}: Val Accuracy = {val_acc * 100:.2f}%")

    # Final test evaluation
    with torch.no_grad():
        test_embeds = model(X_test_tensor)
        test_preds = mahalanobis_predict(test_embeds, means, inv_cov)
        test_acc = (test_preds == y_test_tensor).float().mean().item()
        print(f"\n[MahalanobisNet] Test Accuracy: {test_acc * 100:.2f}%")
        print(classification_report(
            y_test_tensor.cpu().numpy(),
            test_preds.cpu().numpy(),
            target_names=label_encoder.classes_
        ))


# ------------------------------ MAIN ------------------------------- #
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    logging.info("\nLoading aligned audio + phoneme labels...")
    data_m = load_aligned_data(male_audios_dir, textgrids_dir)
    data_f = load_aligned_data(female_audios_dir, textgrids_dir)
    all_data = data_m + data_f
    logging.info(f"Loaded {len(all_data)} aligned utterances.")

    logging.info("\nExtracting LPC features...")
    frame_data = [frames for frames, _ in all_data]
    frame_labels = [labels for _, labels in all_data]

    logging.info(f"Labels: {set(label for labels in frame_labels for label in labels)}")

    lpc_features = Parallel(n_jobs=N_JOBS)(
        delayed(extract_lpc_features)(frames) for frames in tqdm(frame_data, desc="LPC Extraction")
    )
    filtered_features = []
    filtered_labels = []

    for features, labels in zip(lpc_features, frame_labels):
        if features.shape[0] > 0:
            filtered_features.append(features)
            filtered_labels.append(labels)

    all_features = np.vstack(filtered_features)
    all_labels = np.array([lbl for seq in filtered_labels for lbl in seq])

    # Drop unwanted labels
    mask = np.array([label not in DROP_LABELS for label in all_labels])
    all_features = all_features[mask]
    all_labels = all_labels[mask]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(all_labels)
    logging.info(f"Classes: {label_encoder.classes_}")
    logging.info(f"{len(label_encoder.classes_)} unique phonemes")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, y_encoded, test_size=0.2, random_state=42
    )

    # Optional val split
    # FIXME not used?
    val_size = int(X_train.shape[0] * VAL_RATIO)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Optional: apply PCA (improves Mahalanobis stability)
    # pca = PCA(n_components=0.95)
    # X_train = pca.fit_transform(X_train)
    # X_val = pca.transform(X_val)
    # X_test = pca.transform(X_test)

    # Compute means and covariances
    means = {}
    covs = {}
    inv_covs = {}
    for label in np.unique(y_train):
        class_data = X_train[y_train == label]
        mean = np.mean(class_data, axis=0)
        cov = np.cov(class_data, rowvar=False)
        cov += REGULARIZATION * np.eye(cov.shape[0])
        means[label] = mean
        covs[label] = cov
        inv_covs[label] = np.linalg.inv(cov)

    logging.info("\nClassifying test data using Mahalanobis distance...")
    y_pred = mahalanobis_classification(X_test, means, inv_covs)

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"\nAccuracy: {accuracy * 100:.2f}%")
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    compare_mahalanobis_classifiers(X_train, y_train, X_test, y_test, label_encoder)
