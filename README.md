**Sentiment Analysis using BiLSTM (End-to-End NLP Pipeline)**

This project implements a production-grade sentiment analysis system using Bidirectional LSTM (BiLSTM).

It follows a clean ML pipeline architecture with preprocessing, tokenization, training, evaluation, and artifact management.

The goal is to demonstrate industry-standard NLP workflow using Python, TensorFlow/Keras, and Pandas.

## 📓 Jupyter Notebook (Rendered)

If GitHub does not render the notebook properly, you can view it here:

🔗 **Notebook Viewer (nbviewer)**  
https://nbviewer.org/github/Swap1984/Sentiment-Analysis-using-BiLSTM/blob/main/Notebooks/data_loading_processing.ipynb
)

## 📓 Jupyter Notebook (Rendered)

If GitHub does not render the notebook properly, you can view it here:

🔗 **Notebook Viewer (nbviewer)**  
https://nbviewer.org/github/Swap1984/Sentiment-Analysis-using-BiLSTM/blob/main/Notebooks/data_loading_processing.ipynb


**📌 Project Highlights**

End-to-end NLP pipeline (raw data → trained model)

Robust text preprocessing using regex

Tokenization & padding using Keras

Label encoding for sentiment classes

BiLSTM deep learning model

Modular production-style code (src/)

Reproducible training & evaluation

Validation accuracy ~84% and external validation ~93%

RNN_new/

│

├── data/

│   ├── raw/

│   │   ├── twitter_training.csv

│   │   └── twitter_validation.csv

│   │
│   ├── processed/

│   │   ├── train_processed.csv

│   │   └── val_processed.csv

│

├── artifacts/

│   ├── sentiment_model.h5

│   ├── tokenizer.pkl

│   └── label_encoder.pkl

│

├── src/

│   ├── __init__.py

│   ├── config.py

│   ├── preprocess.py

│   ├── text_tokenizer.py

│   ├── model.py

│   ├── train.py

│   ├── evaluate.py

│   └── utils.py

│

├── notebooks/

│   └── exploration.ipynb

│    └── results/

    ├── metrics.txt
    
    ├── classification_report.txt
    
    ├── confusion_matrix.png
    
    ├── training_history.csv
    
    └── final_inference.md


├── README.md

├── requirements.txt

└── .gitignore

**📊 Dataset**

Source: Twitter Sentiment Dataset

Columns used:

text → Input feature (X)

sentiment → Target label (y)

Only the text column is tokenized and fed to the model.

The sentiment column is label-encoded and used as the prediction target


**⚙️ Text Preprocessing**

Steps applied:

Remove URLs

Normalize repeated characters

Clean excessive punctuation

Normalize whitespace

Drop null values

Implemented in::src/preprocess.py


**🔠 Tokenization & Encoding**

Tokenizer: Keras Tokenizer

Vocabulary size: 10,000

Sequence length: 100

Out-of-vocabulary token supported

Label encoding using LabelEncoder

Implemented in::src/text_tokenizer.py

**Artifacts saved:**

tokenizer.pkl

label_encoder.pkl


**🧠 Model Architecture**

Embedding Layer

Bidirectional LSTM

Dropout Regularization

Dense Softmax Output Layer

Implemented in:src/model.py


**🚀 Training Pipeline**

The training script:

Loads processed CSVs

Tokenizes text

Encodes labels

Trains BiLSTM

Saves trained model & artifacts

Run:python src/train.py


**📈 Model Evaluation**

Evaluation includes:

Validation accuracy & loss

Predictions on validation dataset

Run:python src/evaluate.py


**📦 Artifacts Generated**

Stored in artifacts/:

Trained model (.h5)

Tokenizer

Label encoder

These are reused for inference without retraining.

🧪 Environment Setup

Create virtual environment:
python -m venv .venv


Activate:
# Windows
.venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

🏁 How to Run the Project (Order Matters)

python src/preprocess.py

python src/train.py

python src/evaluate.py

**Evaluation matrix**

Classification Report:

              precision    recall  f1-score   support

  Irrelevant       0.92      0.93      0.93       172
    Negative       0.94      0.95      0.94       266
     Neutral       0.95      0.93      0.94       285
    Positive       0.93      0.94      0.94       277

    accuracy                           0.94      1000
   macro avg       0.94      0.94      0.94      1000
weighted avg       0.94      0.94      0.94      1000



🧑‍💻 Author

Swapnil Sudhakar Patil

Electronics Engineer → Data Scientist / GenAI Engineer

Specialized in NLP, Deep Learning, and Production ML Pipelines

📌 Future Improvements

FastAPI inference service

Dockerization

MLflow experiment tracking

Transformer-based models (BERT)

CI/CD pipeline


