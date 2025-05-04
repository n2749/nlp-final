import numpy as np
import tensorflow as tf
from keras import layers, models

# Parameters
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 2


# Sinusoidal positional encoding
def get_positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)  # (1, max_len, d_model)


# Custom layer for adding positional encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = get_positional_encoding(max_len, d_model)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


# Transformer Encoder Block
def transformer_encoder_block(embed_dim, num_heads, ff_dim, dropout=0.1):
    inputs = layers.Input(shape=(None, embed_dim))

    # Multi-Head Self-Attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = layers.Dropout(dropout)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-Forward Network
    ffn = models.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(embed_dim),
    ])
    ffn_output = ffn(out1)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return models.Model(inputs=inputs, outputs=out2)


# Build full Transformer-based model for classification
def build_transformer_classifier():
    inputs = layers.Input(shape=(MAX_LEN,))
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
    x = PositionalEncoding(MAX_LEN, EMBEDDING_DIM)(x)

    # Stack Transformer blocks
    for _ in range(NUM_LAYERS):
        x = transformer_encoder_block(EMBEDDING_DIM, NUM_HEADS, FF_DIM)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


