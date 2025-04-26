import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis

# Directories (update with your actual paths)
male_audios_dir = "/media/lana/ExternalSSD/Faks/Komunikacija covjek stroj/VEPRAD/audio_m"
female_audios_dir = "/media/lana/ExternalSSD/Faks/Komunikacija covjek stroj/VEPRAD/audio_z"
male_transcripts_dir = "/media/lana/ExternalSSD/Faks/Komunikacija covjek stroj/VEPRAD/transkripcije_m01-m11"
female_transcripts_dir = "/media/lana/ExternalSSD/Faks/Komunikacija covjek stroj/VEPRAD/transkripcije_z01-z14"

FRAME_LENGTH = 0.03  # 30 ms
FRAME_OVERLAP = 0.5  # 50% overlap
LPC_ORDER = 12

# TODO define VEPRAD alphabet
alphabet_tokens = {}

# TODO tokenize transcripts
def tokenize_transcript(transcript):
    tokens = []
    for phoneme in transcript:
        if phoneme in alphabet_tokens:
            tokens.append(alphabet_tokens[phoneme])
        else:
            print(f"Phoneme {phoneme} not found in alphabet.")
    return tokens

# Load audio and phoneme transcripts
def load_audio_and_transcripts(audio_dir, transcript_dir):
    data = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                y, sr = librosa.load(audio_path, sr=None)

                # Load transcript (phoneme sequence)
                transcript_path = os.path.join(transcript_dir, file.replace('.wav', '.txt'))
                with open(transcript_path, 'r') as f:
                    phonemes = f.read().strip().split()  # Assuming space-separated phonemes

                data.append((y, sr, phonemes))
    return data

# Segment signal into frames
def frame_signal(y, sr):
    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    return frames

# Compute LPC coefficients for a signal
def lpc_coeffs(y, order=12):
    return librosa.lpc(y, order=order)

# Main
if __name__ == "__main__":
    # Load audio and phoneme transcripts
    audio_data_m = load_audio_and_transcripts(male_audios_dir, male_transcripts_dir)
    audio_data_f = load_audio_and_transcripts(female_audios_dir, female_transcripts_dir)
    audio_data_all = audio_data_m + audio_data_f

    # Frame signal data and phonemes
    frame_data = []
    frame_labels = []

    for y, sr, phonemes in audio_data_all:
        frames = frame_signal(y, sr)
        frame_data.append(frames)

        # Assign phonemes to frames (simplified alignment)
        total_frames = frames.shape[0]
        phoneme_count = len(phonemes)
        
        # Distribute phonemes to frames
        for i in range(total_frames):
            # Approximate: map phonemes to frames
            phoneme_idx = min(int(i * phoneme_count / total_frames), phoneme_count - 1)
            frame_labels.append(phonemes[phoneme_idx])

    # Compute LPC features for each frame
    lpc_features = []
    for frames in frame_data:
        features = np.array([lpc_coeffs(frame, order=LPC_ORDER) for frame in frames])
        lpc_features.append(features)

    X = np.vstack(lpc_features)
    y = np.array(frame_labels)

    # Encode phoneme labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(label_encoder.classes_)  # e.g., ['aa', 'ae', 'ah', ..., 'zh']

    # TODO Create a vocabulary
    tokenized_transcript = []
    vocab = {token: idx for idx, token in enumerate(set(tokenized_transcript))}
    vocab_size = len(vocab)

    # Convert tokens into indices
    token_indices = [vocab[token] for token in tokenized_transcript]

    print(token_indices[:10])

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Build Mahalanobis classifier
    means = {}
    covs = {}

    for label in np.unique(y_train):
        X_label = X_train[y_train == label]
        means[label] = np.mean(X_label, axis=0)
        covs[label] = np.cov(X_label, rowvar=False) + 1e-6 * np.eye(X_label.shape[1])  # Regularization

    y_pred = []

    for x in X_test:
        min_dist = float('inf')
        best_label = None
        for label in means:
            inv_cov = np.linalg.inv(covs[label])
            dist = mahalanobis(x, means[label], inv_cov)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        y_pred.append(best_label)

    y_pred = np.array(y_pred)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")
