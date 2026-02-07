# Inference Demo

This folder contains a minimal demo for running inference with the trained BiLSTM model.

## Files

- `inference_demo.py` - loads `artifacts/tokenizer.pkl`, `artifacts/label_encoder.pkl` and a model file (e.g. `sentiment_model.keras` or `model.h5`) and runs prediction on input text.

## Expected Artifacts

- `artifacts/tokenizer.pkl` - Keras Tokenizer used during training
- `artifacts/label_encoder.pkl` - sklearn LabelEncoder used to map predictions back to labels
- `artifacts/sentiment_model.keras` (or `.h5`) - trained Keras model

## How to Run

1. Generate artifacts by running preprocessing and training scripts:

```bash
python src/preprocess.py
python src/train.py
```

2. Run the demo with a sample text:

```bash
python examples/inference_demo.py "I love this product!"
```

## Notes

- The demo expects artifacts to be present in the top-level `artifacts/` directory. If they are large, keep them out of the repository and publish as release assets or provide a download link.
- If TensorFlow is not installed in your environment, install dependencies from `requirements.txt` before running the demo.

