import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTower(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class UserTower(BaseTower):
    pass


class ItemTower(BaseTower):
    pass


class TwoTowerRecSys(nn.Module):
    def __init__(self, user_dim, item_dim, embedding_dim):
        super().__init__()
        self.user_tower = UserTower(input_dim=user_dim, embedding_dim=embedding_dim)
        self.item_tower = ItemTower(input_dim=item_dim, embedding_dim=embedding_dim)

    def forward(self, user_x, item_x):
        user_embedding = self.user_tower(user_x)
        item_embedding = self.item_tower(item_x)
        return user_embedding, item_embedding


def train(model, user_data, item_data, epochs=10, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(user_data), batch_size):
            user_batch = user_data[i:i+batch_size]
            item_batch = item_data[i:i+batch_size]

            user_embedding, item_embedding = model(user_batch, item_batch)

            similarity = torch.matmul(user_embedding, item_embedding.T)

            labels = torch.arange(similarity.size(0)).to(similarity.device)

            loss = criterion(similarity, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch+1} loss = {loss.item():.4f}")


def generate_data(
    n_samples=1_000, n_features=10, n_classes=5, noise=0.1, seed=8,
) -> torch.Tensor:
    """Generate random user and item data with a number of "latent
    classes", representing unlabeled classes that should be learned by
    a model during training.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features per observation/sample.
        n_classes: Number of latent classes to be preent in the data.
        noise: Scalar multiple to apply to the noise added to each
            class.

    Returns:
        Two [n_samples x n_features] matrices.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    class_roots = torch.randn(n_classes, n_features)

    user_features = []
    item_features = []

    for _ in range(n_samples):
        class_ = np.random.randint(0, n_classes)
        class_root = class_roots[class_]

        user_feature = class_root + noise * torch.randn(n_features)
        item_feature = class_root + noise * torch.randn(n_features)

        user_features.append(user_feature)
        item_features.append(item_feature)
    return torch.stack(user_features), torch.stack(item_features)
