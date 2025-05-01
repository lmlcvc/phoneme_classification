import os
import numpy as np
import librosa

import torch
# from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from scipy.spatial.distance import mahalanobis

from joblib import Parallel, delayed
from tqdm import tqdm
import configparser

# from dataset import VEPRADDataset

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

# Directories
male_audios_dir = config['male_audios_dir']
female_audios_dir = config['female_audios_dir']
male_transcripts_dir = config['male_transcripts_dir']
female_transcripts_dir = config['female_transcripts_dir']

# TODO move all to config.ini
FRAME_LENGTH = 0.03
FRAME_OVERLAP = 0.5

LPC_ORDER = 12
# TODO simple upgrade idea: LPC + MFCC = about 50 features, and works very well for phonemes

N_JOBS = -1

VAL_RATIO = 0.2
BATCH_SIZE = 64


# TODO đ??
alphabet_tokens = {
    'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i',
    'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r',
    's': 's', 't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z',
    'lj': 'L', 'nj': 'N', 'dž': 'D',  # Digraphs
    '~': '~', '^': '^', '}': '}', '{': '{', '`': '`',  # č, ć, dž, š, ž  
    '<sil>': '<sil>', '<uzdah>': '<uzdah>', '<papir>': '<papir>'  # Multi-character tokens
}  

def tokenize_transcript(transcript: str) -> list:
    """
    Tokenize the transcript into valid tokens using the alphabet_tokens dictionary.
    Handles multi-character tokens (<sil>, <uzdah>, <papir>) and digraphs (lj, nj, dž).
    """
    tokens = []
    i = 0
    while i < len(transcript):
        # Handle multi-character tokens
        if transcript[i] == '<':
            end_idx = transcript.find('>', i)
            if end_idx != -1:
                token = transcript[i:end_idx + 1]
                if token in alphabet_tokens:
                    tokens.append(alphabet_tokens[token])
                i = end_idx + 1
                continue
        
        # Handle digraphs (lj, nj, dž)
        if transcript[i:i+2] in alphabet_tokens:
            tokens.append(alphabet_tokens[transcript[i:i+2]])
            i += 2
            continue
        
        # Handle single-character tokens
        if transcript[i] in alphabet_tokens:
            tokens.append(alphabet_tokens[transcript[i]])
        i += 1
    return tokens

def load_audio_and_transcripts(audio_dir, transcript_dir):
    data = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                y, sr = librosa.load(audio_path, sr=None)
                speaker_id = os.path.basename(root)
                transcript_path = os.path.join(transcript_dir, speaker_id, file.replace('.wav', '.txt'))
                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r') as f:
                        transcript = f.read().strip()
                    data.append((y, sr, transcript))
                else:
                    print(f"Transcript not found for {audio_path}")
    return data

def frame_signal(y, sr):
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    return frames

def extract_lpc_features(frames):
    return np.array([librosa.lpc(frame, order=LPC_ORDER) for frame in frames])

def same_seeds(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mahalanobis_classification(X, means, covs):
    preds = []
    for x in X:
        dists = [
            mahalanobis(x, means[label], np.linalg.inv(covs[label]))
            for label in means
        ]
        preds.append(np.argmin(dists))
    return np.array(preds)

if __name__ == "__main__":
    print(f"\nUsing device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Load audio files and transcripts
    print("\nLoading audio files and transcripts...")
    audio_data_m = load_audio_and_transcripts(male_audios_dir, male_transcripts_dir)
    audio_data_f = load_audio_and_transcripts(female_audios_dir, female_transcripts_dir)
    audio_data_all = audio_data_m + audio_data_f
    print(f"Loaded {len(audio_data_all)} audio files with transcripts.")

    ## Preprocess audio data
    print("\nPreprocessing audio data...")
    frame_data = []
    frame_labels = []

    for y, sr, transcript in tqdm(audio_data_all, desc="Framing and labeling"):
        # Tokenize the transcript
        phonemes = tokenize_transcript(transcript)
        
        # Segment the audio signal into frames
        frames = frame_signal(y, sr)
        frame_data.append(frames)
        
        # Assign labels to frames (improved alignment using energy)
        total_frames = frames.shape[0]
        phoneme_count = len(phonemes)
        
        # Estimate energy for each frame
        energies = np.sum(frames**2, axis=1)

        # Define an energy threshold (25th percentile)
        threshold = np.percentile(energies, 25)

        # Identify active (voiced) frames
        active_frames = [i for i, e in enumerate(energies) if e > threshold]

        if phoneme_count == 0:
            print("Warning: empty transcript")
            frame_labels.extend(['<sil>'] * total_frames)
            continue

        # Split active frames into chunks corresponding to phonemes
        if len(active_frames) >= phoneme_count:
            chunks = np.array_split(active_frames, phoneme_count)
        else:
            chunks = [active_frames]  # fallback: treat all active frames as one phoneme

        assigned = set()
        for phoneme, chunk in zip(phonemes, chunks):
            for i in chunk:
                frame_labels.append(phoneme)
                assigned.add(i)

        # Assign <sil> to unvoiced/silence frames
        for i in range(total_frames):
            if i not in assigned:
                frame_labels.append('<sil>')

    # TODO save tokenized transcript

    # Extract LPC features
    lpc_features = Parallel(n_jobs=N_JOBS)(
        delayed(extract_lpc_features)(frames) for frames in tqdm(frame_data, desc="Extracting LPC")
    )

    X = np.vstack(lpc_features)
    y = np.array(frame_labels)
    print(f"Extracted {X.shape[0]} frames with {X.shape[1]} features each.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(label_encoder.classes_)
    print(f"Labels: {len(np.unique(y))} unique phonemes")

    # Split the data into training, validation and test sets
    X, X_test, y, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    percent = int(X.shape[0] * (1 - VAL_RATIO))
    X_train, X_val = X[:percent], X[percent:]
    y_train, y_val = y[:percent], y[percent:]
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Apply PCA
    # pca = PCA(n_components=20)  # Retain 20 principal components
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # TODO use class priors

    # Compute means and covariances
    means = {}
    covs = {}

    for label in np.unique(y_train):
        class_data = X_train[y_train == label]
        means[label] = np.mean(class_data, axis=0)
        cov = np.cov(class_data, rowvar=False)
        cov += 1e-5 * np.eye(cov.shape[0])  # Regularization
        covs[label] = cov

    # Predict and evaluate
    y_pred = mahalanobis_classification(X_val, means, covs)
    val_acc = np.mean(y_pred == y_val)
    print(f"\nValidation accuracy: {val_acc:.4f}")

    y_pred_test = mahalanobis_classification(X_test, means, covs)
    test_acc = np.mean(y_pred_test == y_test)
    print(f"Test accuracy: {test_acc:.4f}")
