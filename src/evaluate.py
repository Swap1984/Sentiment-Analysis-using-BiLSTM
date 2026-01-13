import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

from src.config import RAW_DATA_DIR, ARTIFACTS_DIR
from src.text_tokenizer import TextTokenizer
from src.utils import load_pickle
from src.preprocess import clean_text


def main():
    print("Loading model and artifacts...")

    model = load_model(ARTIFACTS_DIR / "sentiment_model.keras")
    tokenizer = TextTokenizer.load(ARTIFACTS_DIR / "tokenizer.pkl")
    label_encoder = load_pickle(ARTIFACTS_DIR / "label_encoder.pkl")

    val_df = pd.read_csv(
    RAW_DATA_DIR / "twitter_validation.csv",
    header=None,
    names=["tweet_id", "entity", "sentiment", "text"]
)

    val_df = val_df.dropna(subset=["text", "sentiment"])
    val_df["clean_text"] = val_df["text"].apply(clean_text)

    X_val = tokenizer.transform(val_df["clean_text"])
    y_true = label_encoder.transform(val_df["sentiment"])

    y_pred = model.predict(X_val)
    y_pred_labels = y_pred.argmax(axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred_labels,
                                target_names=label_encoder.classes_))


if __name__ == "__main__":
    main()
