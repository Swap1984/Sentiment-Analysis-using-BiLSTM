"""Simple inference demo for the saved BiLSTM model and artifacts.

Usage:
  python examples/inference_demo.py "This product is great!"

If artifacts are missing the script will explain what to generate.
"""
from pathlib import Path
import pickle
import sys

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def find_model_file(artifacts_dir: Path):
    candidates = [
        "sentiment_model.keras",
        "sentiment_model.h5",
        "model.keras",
        "model.h5",
    ]
    for name in candidates:
        p = artifacts_dir / name
        if p.exists():
            return p
    # fallback: any .keras/.h5 in artifacts
    for p in artifacts_dir.glob("*.keras"):
        return p
    for p in artifacts_dir.glob("*.h5"):
        return p
    return None


def main(texts: list[str]):
    if not ARTIFACTS.exists():
        print(
            "No artifacts/ directory found. Run `python src/train.py` and/or "
            "`python src/preprocess.py` to generate artifacts."
        )
        return

    tok_path = ARTIFACTS / "tokenizer.pkl"
    lbl_path = ARTIFACTS / "label_encoder.pkl"
    model_path = find_model_file(ARTIFACTS)

    if not tok_path.exists() or not lbl_path.exists() or model_path is None:
        print("Missing artifacts. Expected:")
        print(f" - {tok_path} (tokenizer)")
        print(f" - {lbl_path} (label encoder)")
        print(f" - model file (e.g. sentiment_model.keras)")
        print("Generate artifacts by running preprocessing and training scripts.")
        return

    if load_model is None:
        print("TensorFlow not available. Install dependencies from requirements.txt")
        return

    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(lbl_path, "rb") as f:
        label_encoder = pickle.load(f)

    model = load_model(model_path)

    # simple tokenization using keras tokenizer (assumes same API used in training)
    sequences = tokenizer.texts_to_sequences(texts)
    # pad to length from config if available
    try:
        from src.config import MAX_LEN

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    except Exception:
        X = sequences

    preds = model.predict(X)
    labels = label_encoder.inverse_transform(preds.argmax(axis=1))
    for t, p, l in zip(texts, preds, labels):
        print("----")
        print("Text:", t)
        print("Predicted label:", l)
        print("Probabilities:", [float(x) for x in p])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    else:
        texts = ["I love this! Amazing product."]
    main(texts)
