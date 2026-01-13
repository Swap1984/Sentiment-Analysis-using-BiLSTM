# Twitter Sentiment Analysis using BiLSTM

## 📌 Overview
This project implements a **Bidirectional LSTM-based sentiment analysis model**
trained on Twitter data to classify tweets into:
- Negative
- Neutral
- Positive

The model achieved:
- **83.9% validation accuracy**
- **93% external validation accuracy**

---

## 🧠 Model Architecture
- Embedding Layer (trainable)
- Bidirectional LSTM
- Dropout Regularization
- Softmax Output Layer

---

## 🗂 Project Structure

---project-name/
│
├── data/
│   ├── raw/          # Original datasets (never modify)
│   └── processed/    # Cleaned, tokenized data
│
├── src/              # All Python source code
│   ├── preprocess.py
│   ├── tokenize.py
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
│
├── models/           # Saved trained models
│
├── notebooks/        # Jupyter notebooks (experiments only)
│
├── requirements.txt  # Libraries
├── README.md         # Project explanation
└── .gitignore


## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/train.py


**Results**
| Dataset             | Accuracy |
| ------------------- | -------- |
| Validation          | 83.9%    |
| External Validation | 93%      |

🛠 Tech Stack

Python

TensorFlow / Keras

Scikit-learn

Pandas

NumPy

📌 Future Improvements

Transformer-based models (BERT)

Emoji-aware embeddings

Real-time sentiment API

👤 Author

Swapnil Sudhakar Patil