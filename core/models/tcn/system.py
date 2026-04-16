"""
VAJRA TCN MODEL V5.0 (Master Architecture)
==========================================
Architectural Updates:
1. Input Attention: Dynamically weights features (e.g., prioritizes VWAP over RSI in volatility).
2. Focal Loss Integration: Integrated directly into the class structure.
3. Clean Architecture: Removed legacy helper classes to prevent namespace pollution.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, Add, Activation,
    GlobalAveragePooling1D, BatchNormalization, MultiHeadAttention,
    LayerNormalization, Reshape, Multiply, Permute
)
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import json

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VAJRA_MODEL")

def focal_loss(gamma=2.5, alpha=0.70):
    """
    Precision Loss function for Class Imbalance (Bull/Bear trends are rare).
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(alpha_weight * focal_weight * bce)
    return loss

class TemporalBlock(tf.keras.layers.Layer):
    """Residual Causal Convolution Block"""
    def __init__(self, n_filters, kernel_size, dilation_rate, dropout_rate=0.4, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.conv1 = Conv1D(filters=self.n_filters, kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate, padding='causal', activation='relu')
        self.bn1 = BatchNormalization()
        self.drop1 = Dropout(self.dropout_rate)
        
        self.conv2 = Conv1D(filters=self.n_filters, kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate, padding='causal', activation='relu')
        self.bn2 = BatchNormalization()
        self.drop2 = Dropout(self.dropout_rate)

        if input_shape[-1] != self.n_filters:
            self.downsample = Conv1D(filters=self.n_filters, kernel_size=1)
        else:
            self.downsample = None
            
        self.add = Add()
        self.act = Activation('relu')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)
        
        res = self.downsample(inputs) if self.downsample else inputs
        return self.act(self.add([x, res]))

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout_rate": self.dropout_rate,
        })
        return config

class TCNTradingModel:
    def __init__(self, sequence_length=64, n_features=54, learning_rate=1e-3):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self):
        inputs = Input(shape=(self.sequence_length, self.n_features))

        # --- FEATURE ATTENTION (The "Architect's" Addition) ---
        # Weights features before temporal processing. 
        # Helps model decide if "RSI" is more important than "VWAP" at this moment.
        # Shape: (batch, seq, features) -> (batch, features) attention -> applied
        feat_attn = Dense(self.n_features, activation='softmax', name='feature_attention')(inputs)
        x_weighted = Multiply()([inputs, feat_attn])
        
        # --- TCN BACKBONE ---
        x = x_weighted
        for i, dilation in enumerate([1, 2, 4, 8]):
            x = TemporalBlock(n_filters=64, kernel_size=3, dilation_rate=dilation)(x)

        # --- TEMPORAL ATTENTION ---
        # Focus on specific time steps (e.g., recent price action vs distant setup)
        attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)
        
        # --- HEADS ---
        pooled = GlobalAveragePooling1D()(x)
        
        # Direction Head (Classification)
        d = Dense(64, activation='relu')(pooled)
        d = Dropout(0.4)(d)
        dir_out = Dense(1, activation='sigmoid', name='direction')(d)
        
        # Volatility Head (Regression)
        v = Dense(64, activation='relu')(pooled)
        v = Dropout(0.4)(v)
        vol_out = Dense(1, activation='softplus', name='volatility')(v)

        self.model = Model(inputs=inputs, outputs=[dir_out, vol_out])
        
        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss={'direction': focal_loss(), 'volatility': 'mse'},
            loss_weights={'direction': 1.0, 'volatility': 0.5},
            metrics={'direction': ['accuracy', 'AUC'], 'volatility': ['mae']}
        )
        return self.model

    def train(self, X_train, y_dir, y_vol, X_val, y_dir_val, y_vol_val, epochs=50, batch_size=32, callbacks=None):
        # Features are now self-standardized by the feature engine.
        # Volatility targets are used raw (annualized volatility is already in a stable range).
        
        self.history = self.model.fit(
            X_train, {'direction': y_dir, 'volatility': y_vol},
            validation_data=(X_val, {'direction': y_dir_val, 'volatility': y_vol_val}),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1
        )
        return self.history

    def predict(self, X):
        # Features are now self-standardized by the feature engine.
        pred_dir, pred_vol = self.model.predict(X, verbose=0)
        return pred_dir, pred_vol

    def save(self, path):
        self.model.save(path + ".keras")
        logger.info(f"Model saved to {path}.keras")

    def load(self, path):
        self.model = tf.keras.models.load_model(path + ".keras", custom_objects={'loss': focal_loss(), 'TemporalBlock': TemporalBlock})
        logger.info(f"Model loaded from {path}.keras")
