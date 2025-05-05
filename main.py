import os
import numpy as np
import librosa
from textgrid import TextGrid
import torch

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial.distance import mahalanobis
from joblib import Parallel, delayed
from tqdm import tqdm
import configparser

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
REGULARIZATION = 1e-6  # Experiment with this value for better results

# ------------------------ PHONEME ALIGNMENT ------------------------ #
def align_phonemes_to_frames(wav_path, grid_path):
    y, sr = librosa.load(wav_path, sr=None)
    frames = librosa.util.frame(y, frame_length=int(FRAME_LENGTH * sr), hop_length=int(FRAME_STEP * sr)).T
    tg = TextGrid.fromFile(grid_path)

    phoneme_tier = next(t for t in tg.tiers if t.name.lower() in ["phones", "phoneme", "phonemes"])
    frame_times = librosa.frames_to_time(np.arange(frames.shape[0]), sr=sr, hop_length=int(FRAME_STEP * sr))

    frame_labels = []
    for t in frame_times:
        label = '<sil>'
        for interval in phoneme_tier.intervals:
            if interval.minTime <= t < interval.maxTime:
                raw_mark = interval.mark.strip()
                if raw_mark == "":
                    label = '<sil>'
                elif raw_mark.lower() in ['spn', 'unk', '???']:
                    label = '<unk>'
                else:
                    label = raw_mark
                break
        frame_labels.append(label)

    return frames, frame_labels

# --------------------------- DATA LOADER --------------------------- #
def load_aligned_data(audio_dir, textgrid_dir):
    data = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue
            audio_path = os.path.join(root, file)
            speaker_id = os.path.basename(root)

            # FIXME inspect why files are missing
            textgrid_path = os.path.join(textgrid_dir, speaker_id, file.replace('.wav', '.TextGrid'))
            if not os.path.exists(textgrid_path):
                print(f"Missing TextGrid for {audio_path}")
                continue

            try:
                frames, labels = align_phonemes_to_frames(audio_path, textgrid_path)
                if frames.shape[0] != len(labels):
                    print(f"Frame-label mismatch: {file}")
                    continue
                data.append((frames, labels))
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return data

# --------------------------- LPC FEATURES --------------------------- #
def extract_lpc_features(frames):
    # Compute LPC coefficients for each frame
    return np.array([librosa.lpc(frame, order=LPC_ORDER) for frame in frames])

# ------------------------- MAHALANOBIS CLF -------------------------- #
def mahalanobis_classification(X, means, covs):
    preds = []
    for x in tqdm(X):
        dists = [
            mahalanobis(x, means[label], np.linalg.inv(covs[label]))
            for label in means
        ]
        preds.append(np.argmin(dists))
    return np.array(preds)

# ------------------------------ MAIN ------------------------------- #
if __name__ == "__main__":
    print(f"\nUsing device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Load data
    print("\nLoading aligned audio + phoneme labels...")
    data_m = load_aligned_data(male_audios_dir, textgrids_dir)
    data_f = load_aligned_data(female_audios_dir, textgrids_dir)
    all_data = data_m + data_f
    print(f"Loaded {len(all_data)} aligned utterances.")

    # Prepare frames and labels
    print("\nExtracting LPC features...")
    frame_data = [frames for frames, _ in all_data]
    frame_labels = [labels for _, labels in all_data]

    lpc_features = Parallel(n_jobs=N_JOBS)(
        delayed(extract_lpc_features)(frames) for frames in tqdm(frame_data)
    )
    all_features = np.vstack(lpc_features)
    all_labels = np.array([lbl for seq in frame_labels for lbl in seq])

    print(f"Feature matrix: {all_features.shape}, Labels: {len(all_labels)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(all_labels)
    print(f"Classes: {label_encoder.classes_}")
    print(f"{len(label_encoder.classes_)} unique phonemes")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, y_encoded, test_size=0.2, random_state=42
    )

    # Optional val split from training set
    val_size = int(X_train.shape[0] * VAL_RATIO)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Class means + covariances
    means = {}
    covs = {}
    for label in np.unique(y_train):
        class_data = X_train[y_train == label]
        means[label] = np.mean(class_data, axis=0)
        
        cov_matrix = np.cov(class_data, rowvar=False)
        covs[label] = cov_matrix + REGULARIZATION * np.eye(cov_matrix.shape[0])

    # Classify the test data
    print("\nClassifying test data using Mahalanobis distance...")
    y_pred = mahalanobis_classification(X_test, means, covs)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
