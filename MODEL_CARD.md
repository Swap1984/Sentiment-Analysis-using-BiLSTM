# Model Card - BiLSTM Twitter Sentiment

## Short Description

Bidirectional LSTM model trained to classify tweets into sentiment categories (Negative, Neutral, Positive). Intended as a demonstration of an end-to-end NLP pipeline: preprocessing to tokenization to training to evaluation to artifacts.

## Authors and Contact

Swapnil Sudhakar Patil - contact via LinkedIn: `https://www.linkedin.com/in/swapnil-patil`

## Intended Use

- Educational and research purposes
- Not intended for high-risk decisions or production use without further validation

## Dataset

- Source: Twitter sentiment data (local CSVs under `Data/Raw_data`)
- Preprocessing: URLs removed, repeated characters normalized, punctuation cleaned, whitespace normalized (see `src/preprocess.py`)
- Note: verify dataset licensing where the raw data was obtained and ensure redistribution is allowed

## Model Architecture and Training

- Embedding layer (trainable)
- Bidirectional LSTM
- Dropout + Dense softmax output
- Tokenizer: Keras Tokenizer (vocab size 10,000, OOV token used)
- Sequence length: 100
- Training: see `src/train.py` for hyperparameters

## Evaluation Metrics

- Reported validation accuracy: ~83.9%
- External validation accuracy: ~93%
- Detailed classification report and confusion matrix are available in `Notebooks/results` when present

## Limitations and Caveats

- Trained on Twitter data; may not generalize to other domains (news, reviews)
- Possible bias in dataset (offensive or demographic skew). Use caution in sensitive deployments.

## How to Use (Example)

1. Generate artifacts (preprocess + train):

```bash
python src/preprocess.py
python src/train.py
```

2. Run quick inference demo:

```bash
python examples/inference_demo.py "I love this product"
```

## Artifacts

- `artifacts/tokenizer.pkl`
- `artifacts/label_encoder.pkl`
- `artifacts/sentiment_model.keras` (or `.h5`)

## License

Model and code: MIT License (see `LICENSE`).

