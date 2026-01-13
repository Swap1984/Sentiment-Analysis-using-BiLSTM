from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

from src.config import MAX_WORDS, MAX_LEN


def build_bilstm_model(num_classes: int):
    model = Sequential([
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN
        ),
        Bidirectional(LSTM(
            64,
            dropout=0.3,
            recurrent_dropout=0.3
        )),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
