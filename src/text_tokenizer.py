import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.config import MAX_LEN


class TextTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            tokenizer = pickle.load(f)
        return TextTokenizer(tokenizer)
