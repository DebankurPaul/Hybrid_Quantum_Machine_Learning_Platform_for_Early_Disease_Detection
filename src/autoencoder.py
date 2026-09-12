"""
autoencoder.py
==============
Multi-Vertical PyTorch Deep Autoencoder Factory for non-linear dimensionality reduction.
Compresses diverse clinical feature spaces into an identical 8-dimensional latent vector:
  1. WDBC Breast Cancer    : 30 features -> 16 -> 8 bottleneck
  2. UCI Heart Disease     : 13 features -> 10 -> 8 bottleneck
  3. Golub Leukemia Microarray : 7,129 genes -> 256 -> 32 -> 8 bottleneck (Dropout=0.3, BatchNorm)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")


class BaseAutoencoder(nn.Module):
    def encode(self, x):
        return self.encoder(x)


class WDBCAutoencoder(BaseAutoencoder):
    """30 -> 16 -> 8 -> 16 -> 30 Deep Autoencoder for Breast Cytology Descriptors."""
    def __init__(self, input_dim=30, bottleneck_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Linear(16, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return reconstructed, bottleneck


class HeartAutoencoder(BaseAutoencoder):
    """13 -> 10 -> 8 -> 10 -> 13 Deep Autoencoder for Clinical EHR Telemetry."""
    def __init__(self, input_dim=13, bottleneck_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.LeakyReLU(0.1),
            nn.Linear(10, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 10),
            nn.LeakyReLU(0.1),
            nn.Linear(10, input_dim)
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return reconstructed, bottleneck


class GenomicAutoencoder(BaseAutoencoder):
    """64 -> 32 -> 8 -> 32 -> 64 Deep Autoencoder for Filtered Microarray Genomics."""
    def __init__(self, input_dim=64, bottleneck_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.GELU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return reconstructed, bottleneck


def get_autoencoder(dataset_key="wdbc", input_dim=30, bottleneck_dim=8):
    """Factory function instantiating the appropriate PyTorch Autoencoder for a given dataset."""
    dataset_key = dataset_key.lower().strip()
    if dataset_key == "wdbc":
        return WDBCAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
    elif dataset_key in ["heart", "cardiology"]:
        return HeartAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
    elif dataset_key in ["leukemia", "genomics", "golub"]:
        return GenomicAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
    else:
        return WDBCAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)


def train_autoencoder(X_tr_sc, X_te_sc, dataset_key="wdbc", bottleneck_dim=8, epochs=150, batch_size=32, lr=1e-3, seed=42):
    """
    Train PyTorch Autoencoder on standardized features for any clinical dataset.
    
    Returns:
        model          : Trained PyTorch Autoencoder
        X_tr_q_ae      : Bottleneck features mapped to [-pi, pi] for train set
        X_te_q_ae      : Bottleneck features mapped to [-pi, pi] for test set
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = X_tr_sc.shape[1]
    X_tr_t = torch.FloatTensor(X_tr_sc)
    X_te_t = torch.FloatTensor(X_te_sc)
    
    effective_batch = min(batch_size, len(X_tr_sc))
    dataset = torch.utils.data.TensorDataset(X_tr_t, X_tr_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=effective_batch, shuffle=True)
    
    model = get_autoencoder(dataset_key=dataset_key, input_dim=input_dim, bottleneck_dim=bottleneck_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print(f"\n[PyTorch Autoencoder - {dataset_key.upper()}] Training {input_dim} -> {bottleneck_dim} bottleneck...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            
        if epoch % 50 == 0 or epoch == 1:
            avg_loss = total_loss / len(X_tr_sc)
            print(f"  Epoch {epoch:3d}/{epochs} | Reconstruction MSE Loss: {avg_loss:.6f}")
            
    # Extract bottleneck representations
    model.eval()
    with torch.no_grad():
        _, tr_bottleneck = model(X_tr_t)
        _, te_bottleneck = model(X_te_t)
        
    tr_bn_np = tr_bottleneck.numpy()
    te_bn_np = te_bottleneck.numpy()
    
    # Angle normalize to [-pi, pi] for quantum gate encoding
    def angle_normalize(X):
        return np.tanh(X) * np.pi
        
    X_tr_q_ae = angle_normalize(tr_bn_np)
    X_te_q_ae = angle_normalize(te_bn_np)
    
    print(f"  [{dataset_key.upper()} Autoencoder] Extraction complete. Angle range: [{X_tr_q_ae.min():.3f}, {X_tr_q_ae.max():.3f}]")
    return model, X_tr_q_ae, X_te_q_ae
