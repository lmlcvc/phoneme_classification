import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from joblib import Parallel, delayed
from tqdm import tqdm

# Directories
male_audios_dir = "/media/lana/ExternalSSD/Faks/Komunikacija covjek stroj/VEPRAD/audio_m"
female_audios_dir = "/media/lana/ExternalSSD/Faks/Komunikacija covjek stroj/VEPRAD/audio_z"
male_transcripts_dir = "/media/lana/ExternalSSD/Faks/Komunikacija covjek stroj/VEPRAD/transkripcije_m01-m11"
female_transcripts_dir = "/media/lana/ExternalSSD/Faks/Komunikacija covjek stroj/VEPRAD/transkripcije_z01-z14"

FRAME_LENGTH = 0.03
FRAME_OVERLAP = 0.5
LPC_ORDER = 12
N_JOBS = -1

# TODO dž, đ??
alphabet_tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 
                   'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
                   's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                   'L', 'N'                                  # lj, nj
                   '~', '^', '{', '`',                       # č, ć, đ, dž, š, ž  
                   '<sil>', '<uzdah>',                       # silence, uzdah
                   'papir'
                   ]       

def tokenize_transcript(transcript: str) -> list:
    tokens = []
    for char in transcript:
        if char in alphabet_tokens:
            tokens.append(char)
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

if __name__ == "__main__":
    audio_data_m = load_audio_and_transcripts(male_audios_dir, male_transcripts_dir)
    audio_data_f = load_audio_and_transcripts(female_audios_dir, female_transcripts_dir)
    audio_data_all = audio_data_m + audio_data_f
    print(f"Loaded {len(audio_data_all)} audio files with transcripts.")

    print(audio_data_all[0][2])
    print(tokenize_transcript(audio_data_all[0][2]))

    import sys
    sys.exit

    ###

    frame_data = []
    frame_labels = []

    for y, sr, phonemes in tqdm(audio_data_all, desc="Framing and labeling"):
        frames = frame_signal(y, sr)
        frame_data.append(frames)
        total_frames = frames.shape[0]
        phoneme_count = len(phonemes)
        for i in range(total_frames):
            phoneme_idx = min(int(i * phoneme_count / total_frames), phoneme_count - 1)
            frame_labels.append(phonemes[phoneme_idx])

    lpc_features = Parallel(n_jobs=N_JOBS)(
        delayed(extract_lpc_features)(frames) for frames in tqdm(frame_data, desc="Extracting LPC")
    )

    X = np.vstack(lpc_features)
    y = np.array(frame_labels)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(label_encoder.classes_)

    # TODO Create a vocabulary
    tokenized_transcript = []
    vocab = {token: idx for idx, token in enumerate(set(tokenized_transcript))}
    vocab_size = len(vocab)

    token_indices = [vocab[token] for token in tokenized_transcript]
    print(token_indices[:10])

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    means = {}
    covs_inv = {}

    # FIXME this trains only 30 intances because y_train is now phonemes
    for label in tqdm(np.unique(y_train), desc="Training Mahalanobis"):
        X_label = X_train[y_train == label]
        means[label] = np.mean(X_label, axis=0)
        cov = np.cov(X_label, rowvar=False) + 1e-6 * np.eye(X_label.shape[1])
        covs_inv[label] = np.linalg.inv(cov)

    all_means = np.stack([means[label] for label in sorted(means.keys())])
    all_inv_covs = np.stack([covs_inv[label] for label in sorted(covs_inv.keys())])

    y_pred = []

    for x in tqdm(X_test, desc="Predicting"):
        distances = [
            mahalanobis(x, all_means[i], all_inv_covs[i])
            for i in range(len(all_means))
        ]
        y_pred.append(np.argmin(distances))

    y_pred = np.array(y_pred)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")
