import pickle
from tensorflow.keras.callbacks import EarlyStopping

from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR, EPOCHS, BATCH_SIZE
from src.model import build_bilstm_model


def main():
    print("Loading processed data...")

    X_train = pickle.load(open(PROCESSED_DATA_DIR / "X_train.pkl", "rb"))
    X_val = pickle.load(open(PROCESSED_DATA_DIR / "X_val.pkl", "rb"))
    y_train = pickle.load(open(PROCESSED_DATA_DIR / "y_train.pkl", "rb"))
    y_val = pickle.load(open(PROCESSED_DATA_DIR / "y_val.pkl", "rb"))

    num_classes = len(set(y_train))

    model = build_bilstm_model(num_classes)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop]
    )

    model.save(ARTIFACTS_DIR / "sentiment_model.keras")
    print("Training completed and model saved.")


if __name__ == "__main__":
    main()
