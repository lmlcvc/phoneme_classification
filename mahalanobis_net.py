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
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.embedding = nn.Linear(hidden_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: (1, batch, hidden_dim)
        hn = hn.squeeze(0)         # -> (batch, hidden_dim)
        hn = self.bn(hn)
        hn = self.dropout(hn)
        embed = self.embedding(hn)
        logits = self.classifier(embed)
        return logits, embed



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
