import torch
import torch.nn as nn
import torch.nn.functional as F


class MahalanobisNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super(MahalanobisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


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


def mahalanobis_predict(embeddings, class_means, inv_cov):
    dists = []
    for mean in class_means:
        diff = embeddings - mean.unsqueeze(0)
        d = torch.einsum('bi,ij,bj->b', diff, inv_cov, diff)
        dists.append(d)
    dists = torch.stack(dists, dim=1)
    return torch.argmin(dists, dim=1)
