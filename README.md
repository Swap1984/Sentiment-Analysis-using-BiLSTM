# Sentiment Analysis Using BiLSTM

An end-to-end Twitter sentiment analysis pipeline using a Bidirectional LSTM. It demonstrates data cleaning, tokenization, training, evaluation, and artifact management with reproducible scripts.

## Notebook

If GitHub does not render the notebook properly, view it here:

- Notebook Viewer (nbviewer):
  `https://nbviewer.org/url/https://raw.githubusercontent.com/Swap1984/Sentiment-Analysis-using-BiLSTM/main/Notebooks/Sentiment_Analysis_BiLSTM.ipynb`

## Highlights

- End-to-end NLP pipeline (raw data to trained model)
- Robust text preprocessing using regex
- Tokenization and padding using Keras
- Label encoding for sentiment classes
- BiLSTM deep learning model
- Modular production-style code in `src/`
- Reproducible training and evaluation
- Validation accuracy around 84% and external validation around 93%

## Quickstart

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Preprocess, train, and evaluate:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

Note: training may take several minutes depending on hardware. Use a small subset for quick tests.

## Results

- Validation accuracy: ~83.9%
- External validation: ~93%

Confusion matrix, training curves, and classification reports are available in `Notebooks/results` and `artifacts/` when produced.

## Project Layout

- `src/` - preprocessing, tokenizer, model, training, evaluation
- `Data/` - raw and processed CSVs
- `Notebooks/` - experiments and EDA
- `artifacts/` - saved model, tokenizer, label encoder
- `tests/` - unit tests

## Reproducibility and Artifacts

- Artifacts are stored in `artifacts/` (tokenizer.pkl, label_encoder.pkl, model.keras).
- If artifacts are large they are excluded from the repo; consider publishing them as release assets or storing in cloud storage and linking from this README.

## Contributing

Small contributions and issues are welcome - see `CONTRIBUTING.md` for details.

## Contact

Swapnil Sudhakar Patil - LinkedIn: `https://www.linkedin.com/in/swapnil-patil`

