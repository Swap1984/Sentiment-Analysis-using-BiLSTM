import pandas as pd
import re
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    MAX_WORDS,
    MAX_LEN
)

# ----------------------------
# Text cleaning function
# ----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"!{2,}", " ! ", text)
    text = re.sub(r"\?{2,}", " ? ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Loading training data...")
    train_df = pd.read_csv(
    RAW_DATA_DIR / "twitter_training.csv",
    header=None,
    names=["tweet_id", "entity", "sentiment", "text"]
)


    # keep only required columns
    train_df = train_df.dropna(subset=["text", "sentiment"])
    train_df["clean_text"] = train_df["text"].apply(clean_text)

    X = train_df["clean_text"]
    y = train_df["sentiment"]

    # ----------------------------
    # Train / validation split
    # ----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ----------------------------
    # Label encoding (TARGET)
    # ----------------------------
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # ----------------------------
    # Tokenization (INPUT)
    # ----------------------------
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    X_val_pad = pad_sequences(
        X_val_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    with open(ARTIFACTS_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # ----------------------------
    # Save processed arrays
    # ----------------------------
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pickle.dump(X_train_pad, open(PROCESSED_DATA_DIR / "X_train.pkl", "wb"))
    pickle.dump(X_val_pad, open(PROCESSED_DATA_DIR / "X_val.pkl", "wb"))
    pickle.dump(y_train_enc, open(PROCESSED_DATA_DIR / "y_train.pkl", "wb"))
    pickle.dump(y_val_enc, open(PROCESSED_DATA_DIR / "y_val.pkl", "wb"))

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
