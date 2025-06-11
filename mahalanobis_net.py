from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class MahalanobisNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, n_classes=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)  
        self.bn = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        embed = self.encoder(x)
        logits = self.classifier(embed)
        return logits, embed
    

class MahalanobisRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, hidden_dim=64, n_classes=None):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim*2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.norm = nn.LayerNorm(hidden_dim*2)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_dim, n_classes)

        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [], 
            'val_loss': []  
        }

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: (num_layers * num_directions, batch, hidden_dim)
        hn = hn.view(2, 2, x.size(0), self.lstm.hidden_size)  # (layers, directions, batch, hidden)
        hn_fwd = hn[-1, 0]  # last layer forward
        hn_bwd = hn[-1, 1]  # last layer backward
        hn_cat = torch.cat((hn_fwd, hn_bwd), dim=1)  # (batch, hidden_dim*2)
        hn_cat = self.norm(hn_cat)
        hn_cat = self.dropout(hn_cat)
        embed = self.embedding(hn_cat)
        logits = self.classifier(embed)
        return logits, embed
    
    def plot_training(self, history):
        epochs = range(1, len(history['train_loss']) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='train loss')
        plt.plot(epochs, history['val_loss'], label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], label='train acc')
        plt.plot(epochs, history['val_acc'], label='val acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy over epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_progress.png")

def compute_class_stats(embeddings, labels, n_classes, eps=1e-6):
    means = []
    for c in range(n_classes):
        class_embeds = embeddings[labels == c]
        means.append(class_embeds.mean(dim=0))
    means = torch.stack(means)

    # Shared covariance matrix
    centered = torch.cat([embeddings[labels == c] - means[c] for c in range(n_classes)], dim=0)
    cov = torch.matmul(centered.T, centered) / centered.shape[0]
    cov += eps * torch.eye(cov.size(0), device=cov.device)
    inv_cov = torch.inverse(cov)
    return means, inv_cov


def mahalanobis_scores(embeddings, class_means, inv_cov):
    dists = []
    for mean in class_means:
        diff = embeddings - mean.unsqueeze(0)
        d = torch.einsum('bi,ij,bj->b', diff, inv_cov, diff) # computes the Mahalanobis distance for all samples in the batch to a given class mean
        dists.append(d)
    dists = torch.stack(dists, dim=1)
    return -dists # Negative because we want to minimize the distance
