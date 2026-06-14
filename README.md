# 🧠 EmotionSense AI

Emotion classification system built using Natural Language Processing (NLP), Logistic Regression, and Streamlit.

**Live Demo:**  
https://supremeinferno-text-detection-prediction-app.streamlit.app

---

## Overview

EmotionSense AI is a text classification project that identifies the underlying emotion expressed in a piece of text.

The model uses a CountVectorizer for feature extraction and a Logistic Regression classifier for prediction. Users can enter text through a Streamlit interface and receive the predicted emotion in real time.

Supported emotion classes:

- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

---

## Screenshots

### Application Interface

<p align="center">
  <img src="assets/Screenshot1.png" width="900">
</p>

### Prediction Example

<p align="center">
  <img src="assets/Screenshot2.png" width="900">
</p>

---

## Dataset

The model was trained on a labeled emotion dataset containing text samples mapped to six emotion categories.

### Target Classes

| Label |
|---------|
| Joy |
| Sadness |
| Anger |
| Fear |
| Love |
| Surprise |

---

## Methodology

### Text Preprocessing

- Text normalization
- Lowercase conversion
- Feature extraction using CountVectorizer

### Model Training

- Train-test split
- Logistic Regression classifier
- Model serialization using Joblib

### Prediction Pipeline

```text
Input Text
    ↓
CountVectorizer
    ↓
Feature Vector
    ↓
Logistic Regression
    ↓
Emotion Prediction
```

---

## Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- Streamlit
- Joblib

---

## Project Structure

```text
text-detection-prediction/
│
├── app.py
├── project.ipynb
├── train.txt
├── Count_Vectorizer.joblib
├── LR1.joblib
├── requirements.txt
├── README.md
│
└── assets/
    ├── dashboard.png
    └── prediction.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/supremeinferno/text-detection-prediction.git
```

Move into the project directory:

```bash
cd text-detection-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Work

- TF-IDF feature extraction
- Deep learning-based text classification
- Transformer-based models (BERT)
- Emotion probability visualization
- Multilingual support

---

## Author

**Pranav Garg**

GitHub: https://github.com/supremeinferno
