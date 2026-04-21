import os
import numpy as np
import kagglehub
import tensorflow as tf



class SoftMarginSVM:
    """
    Soft-Margin Linear SVM toi uu bang SGD:
      - Gradient clipping: tranh NaN/Inf khi lr lon
      - Sample weights: xu ly class imbalance
      - LR decay: hoi tu on dinh
    """

    def __init__(self, C=1.0, lr=0.01, n_epochs=30, batch_size=64,
                 lr_decay=0.95, clip_norm=5.0, random_state=42):
        """
        C         : he so soft-margin
        lr        : learning rate
        n_epochs  : so epoch
        batch_size: kich thuoc mini-batch
        lr_decay  : he so giam lr (lr *= lr_decay moi epoch)
        clip_norm : nguong gradient clipping (tranh explode)
        """
        self.C            = C
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.lr_decay     = lr_decay
        self.clip_norm    = clip_norm
        self.random_state = random_state
        self.w            = None
        self.b            = 0.0
        self.train_losses = []
        self.val_losses   = []

    def _hinge_loss(self, X, y, sample_weights=None):
        margins = y * (X @ self.w + self.b)
        hinge   = np.maximum(0, 1 - margins)
        if sample_weights is not None:
            hinge = hinge * sample_weights
        l2_reg = 0.5 * (1.0 / self.C) * np.dot(self.w, self.w)
        return l2_reg + np.mean(hinge)

    def fit(self, X_train, y_train, sample_weights=None,
            X_val=None, y_val=None):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X_train.shape
        lam        = 1.0 / self.C
        current_lr = self.lr

        if sample_weights is None:
            sample_weights = np.ones(n_samples)

        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0

        for epoch in range(1, self.n_epochs + 1):
            idx     = rng.permutation(n_samples)
            X_shuf  = X_train[idx]
            y_shuf  = y_train[idx]
            sw_shuf = sample_weights[idx]

            for start in range(0, n_samples, self.batch_size):
                Xb  = X_shuf [start: start + self.batch_size]
                yb  = y_shuf [start: start + self.batch_size]
                swb = sw_shuf[start: start + self.batch_size]

                margins = yb * (Xb @ self.w + self.b)
                mask    = margins < 1

                dw = lam * self.w
                db = 0.0
                if mask.any():
                    weighted_grad = swb[mask, None] * yb[mask, None] * Xb[mask]
                    dw -= np.mean(weighted_grad, axis=0)
                    db -= np.mean(swb[mask] * yb[mask])

                # --- GRADIENT CLIPPING: gioi han norm cua dw ---
                dw_norm = np.linalg.norm(dw)
                if dw_norm > self.clip_norm:
                    dw = dw * self.clip_norm / dw_norm

                self.w -= current_lr * dw
                self.b -= current_lr * db

            # LR decay
            current_lr *= self.lr_decay

            train_loss = self._hinge_loss(X_train, y_train, sample_weights)
            self.train_losses.append(train_loss)

            val_info = ""
            if X_val is not None:
                val_loss = self._hinge_loss(X_val, y_val)
                self.val_losses.append(val_loss)
                val_info = f"  |  Val Loss: {val_loss:.4f}"

            # Canh bao neu xuat hien NaN
            if np.isnan(train_loss):
                print(f"[CANH BAO] NaN tai epoch {epoch}! Dung training.")
                break

            print(f"Epoch [{epoch:>2}/{self.n_epochs}]  "
                  f"LR: {current_lr:.6f}  "
                  f"Train Loss: {train_loss:.4f}{val_info}")

        return self

    def predict(self, X):
        scores = X @ self.w + self.b
        return np.where(scores >= 0, 1, -1)

    def decision_function(self, X):
        return X @ self.w + self.b

