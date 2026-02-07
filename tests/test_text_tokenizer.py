import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from src.text_tokenizer import TextTokenizer
from src.config import MAX_LEN


def test_texttokenizer_transform_shape():
    texts = ["hello world", "this is a test"]
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    tt = TextTokenizer(tokenizer)
    arr = tt.transform(texts)
    assert arr.shape == (len(texts), MAX_LEN)
    assert np.issubdtype(arr.dtype, np.integer)
