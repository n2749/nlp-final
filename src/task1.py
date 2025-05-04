import time
from matplotlib import pyplot as plt
from keras.datasets import imdb
from keras import layers, Sequential, Model
from keras.layers import Embedding, Bidirectional, Dense, LSTM, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, Input
from keras.preprocessing.sequence import pad_sequences

TRAIN_ACC_FILENAME = "img/task1-train-acc.png"
VAL_ACC_FILENAME = "img/task1-val-acc.png"
VOCAB_SIZE = 10000   # Only use top 10k most frequent words
MAX_LEN = 200        # Max review length (in words)
EMBEDDING_DIM = 64   # Size of word vector embeddings


def get_dataset():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
    return x_train, y_train, x_test, y_test


def get_blstm():
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_transformer():
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(inputs)

    # Positional Encoding (optional for short inputs, or can add learnable)
    attention_output = MultiHeadAttention(num_heads=2, key_dim=EMBEDDING_DIM)(x, x)
    attention_output = Dropout(0.1)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attention_output)

    ffn = Sequential([
        Dense(128, activation='relu'),
        Dense(EMBEDDING_DIM),
    ])
    ffn_output = ffn(out1)
    ffn_output = Dropout(0.1)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    x = GlobalAveragePooling1D()(out2)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def plot(blstm_history, tran_history):
    plt.figure()
    plt.plot(blstm_history.history['accuracy'])
    plt.plot(tran_history.history['accuracy'])
    plt.title('train accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['blstm', 'transformer'], loc='upper left')
    plt.savefig(TRAIN_ACC_FILENAME)

    plt.figure()
    plt.plot(blstm_history.history['val_accuracy'])
    plt.plot(tran_history.history['val_accuracy'])
    plt.title('val accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['blstm', 'transformer'], loc='upper left')

    plt.savefig(VAL_ACC_FILENAME)


def main():
    x_train, y_train, x_test, y_test = get_dataset()
    # measure time
    blstm = get_blstm()
    transformer = get_transformer()


    t1 = time.time()
    blstm_history = blstm.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    t2 = time.time()
    transformer_history = transformer.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    t3 = time.time()

    print(f"blstm fit time: {t2 - t1}")
    print(f"transformer fit time: {t3 - t2}")

    plot(blstm_history, transformer_history)


if __name__ == "__main__":
    main()
