import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from keras.datasets import imdb
from keras import layers, Sequential, Model
from keras.layers import Embedding, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D, Input
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

PLOT_FILENAME = "img/task2-att-scores"
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 64

# Custom Self-Attention Layer
class CustomSelfAttention(layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        weights = self.softmax(scores)
        output = tf.matmul(weights, value)
        return output, weights

def get_dataset():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
    word_index = imdb.get_word_index()
    index_word = {v + 3: k for k, v in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    index_word[3] = "<UNUSED>"
    return x_train, y_train, x_test, y_test, index_word

def get_transformer_with_attention():
    inputs = Input(shape=(MAX_LEN,), name="inputs")
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, name="embedding")(inputs)

    attention_layer = CustomSelfAttention(EMBEDDING_DIM)
    attention_output, attention_scores = attention_layer(x)
    attention_output = Dropout(0.1)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attention_output)

    ffn = Sequential([
        Dense(128, activation='relu'),
        Dense(EMBEDDING_DIM),
    ])
    ffn_output = ffn(out1)
    ffn_output = Dropout(0.1)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    x_pooled = GlobalAveragePooling1D()(out2)
    x_pooled = Dropout(0.1)(x_pooled)
    outputs = Dense(1, activation='sigmoid')(x_pooled)

    model = Model(inputs=inputs, outputs=outputs)
    attention_extractor = Model(inputs=inputs, outputs=attention_scores)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, attention_extractor

def plot_attention(attention_scores, token_ids, index_word):
    attention_matrix = attention_scores[0]  # (seq_len, seq_len)
    tokens = [index_word.get(i, '?') for i in token_ids[:len(attention_matrix)]]

    # Filter out <PAD> tokens
    filtered_indices = [i for i, token in enumerate(tokens) if token != "<PAD>"]
    filtered_tokens = [tokens[i] for i in filtered_indices]
    filtered_matrix = attention_matrix[np.ix_(filtered_indices, filtered_indices)]

    plt.figure(figsize=(12, 10))
    sns.heatmap(filtered_matrix, cmap='viridis', xticklabels=filtered_tokens, yticklabels=filtered_tokens)
    plt.title("Attention Heatmap (Filtered)")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_FILENAME)


def main():
    x_train, y_train, x_test, y_test, index_word = get_dataset()
    transformer, attention_extractor = get_transformer_with_attention()

    transformer_history = transformer.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.1)

    sample_input = x_test[:1]  # First test sample
    attention_scores = attention_extractor.predict(sample_input)
    plot_attention(attention_scores, token_ids=sample_input[0], index_word=index_word)

if __name__ == "__main__":
    main()

