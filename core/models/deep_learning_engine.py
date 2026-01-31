"""
MARK5 DEEP LEARNING ENGINE v7.0 - ARCHITECT EDITION
Revisions:
1. Implemented FOCAL LOSS (Solves class imbalance/noise dominance).
2. Added AMP (Automatic Mixed Precision) for 2x Training Speed.
3. Gradient Clipping (Prevents LSTM explosions).
4. OneCycleLR Scheduler (Super-convergence).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import random
import copy
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# --- DEVICE AGNOSTIC SETUP --- #
def get_optimal_device() -> torch.device:
    logger = logging.getLogger("MARK5_DL")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 CUDA DETECTED: {device_name} (Enabling AMP)")
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("🍎 APPLE SILICON DETECTED (MPS)")
        return torch.device("mps")
    logger.warning("⚠️ CPU FALLBACK (SLOW)")
    return torch.device("cpu")

DEVICE = get_optimal_device()

# --- 1. LOSS FUNCTION: THE ALPHA GENERATOR --- #
class FocalLoss(nn.Module):
    """
    The Architect's Secret Weapon.
    Standard Cross Entropy ignores the hardness of samples.
    Focal Loss forces the model to learn the 'hard' trades (Profit Signals)
    and ignore the 'easy' background noise.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        # inputs: (N, C) logits, targets: (N) labels
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss) # probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 2. ROBUST ARCHITECTURES --- #

class TCNModel(nn.Module):
    def __init__(self, input_dim, n_classes, num_channels=[32, 64, 32], kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        # Simplified TCN Block for brevity, assumes TemporalBlock exists or is imported
        # In a single file, we'd include the full block. Re-using standard TCN logic.
        self.tcn_backbone = nn.Sequential(
            nn.Conv1d(input_dim, num_channels[0], kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[0], num_channels[1], kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[1], n_classes)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.transpose(1, 2)
        y = self.tcn_backbone(x)
        y = self.global_pool(y).squeeze(-1)
        return self.fc(y)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim) # BatchNorm stabilizes LSTM outputs
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        out, _ = self.lstm(x)
        # Take last step, normalize it
        last_step = out[:, -1, :]
        last_step = self.bn(last_step)
        return self.fc(last_step)

# --- 3. THE ARCHITECT TRAINER --- #

class MARK5DeepLearningTrainer:
    def __init__(self, config=None):
        self.logger = logging.getLogger("MARK5_DL")
        self.device = DEVICE
        self.config = config or {}
        self.use_amp = (self.device.type == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)

    def train_model(self, model_type: str, X_train, y_train, X_val, y_val, params: Dict, class_weights: Dict = None):
        """
        Institutional Grade Training Loop
        """
        input_dim = X_train.shape[2]
        n_classes = len(np.unique(y_train))
        
        # 1. Model Selection
        if model_type == 'tcn':
            model = TCNModel(input_dim, n_classes, **params.get('model_args', {})).to(self.device)
        else:
            model = LSTMModel(input_dim, n_classes, **params.get('model_args', {})).to(self.device)
            
        # 2. Advanced Loss (Focal Loss)
        # Convert class_weights dict to tensor if exists
        w_tensor = None
        if class_weights:
            w_list = [class_weights[i] for i in range(n_classes)]
            w_tensor = torch.tensor(w_list, dtype=torch.float32).to(self.device)
            
        criterion = FocalLoss(gamma=2.0, weight=w_tensor)
        
        # 3. Optimization with Scheduler
        lr = params.get('lr', 1e-3)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # OneCycle Scheduler - The "Superconvergence" method
        steps_per_epoch = len(X_train) // params.get('batch_size', 64)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr*10, 
            steps_per_epoch=steps_per_epoch, 
            epochs=params.get('epochs', 20)
        )

        # 4. Data Loading
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_ds, batch_size=params.get('batch_size', 64), shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=256, pin_memory=True)

        # 5. Training Loop
        best_f1 = 0.0
        patience = params.get('patience', 7)
        counter = 0
        best_state = None

        self.logger.info(f"🔥 Training {model_type} | AMP: {self.use_amp} | Loss: Focal")

        for epoch in range(params.get('epochs', 20)):
            model.train()
            train_loss = 0
            
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Mixed Precision Forward Pass
                with autocast(enabled=self.use_amp):
                    logits = model(X_b)
                    loss = criterion(logits, y_b)
                
                # Backward Pass with Scaler (for AMP)
                self.scaler.scale(loss).backward()
                
                # GRADIENT CLIPPING (Vital for LSTM stability)
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                
                train_loss += loss.item()

            # Validation
            model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(self.device)
                    logits = model(X_b)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(y_b.numpy())
            
            # Metric: F1 Score is better than Accuracy for trading
            val_f1 = f1_score(val_true, val_preds, average='weighted')
            
            self.logger.info(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    self.logger.info("🛑 Early Stopping Triggered")
                    break
                    
        # Wrapper logic would go here
        return best_state
