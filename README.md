# 📧 Email Spam Detection using NLP & Machine Learning

An NLP-based machine learning application developed to automatically classify emails as **Spam** or **Not Spam** using **TF-IDF Vectorization** and **Naive Bayes** classifier. The system preprocesses email text, extracts important features, and predicts whether an email is malicious or genuine.

---

## 🚀 Project Overview

Email spam is one of the most common security threats. This project aims to build an intelligent spam detection system that:

- Automatically filters unwanted emails  
- Reduces manual effort in checking mails  
- Provides fast and accurate classification  
- Uses Natural Language Processing techniques

---

## 🛠 Tools & Technologies

- **Programming Language:** Python  
- **Libraries:**  
  - Scikit-learn  
  - Pandas  
  - NumPy  
  - NLTK  
- **Techniques Used:**  
  - TF-IDF Vectorization  
  - Naive Bayes Classification  
  - Text Preprocessing (Tokenization, Stop-word removal, Cleaning)

---

## 📌 Features

✔ Preprocessing of raw email text using NLP  
✔ Feature extraction using TF-IDF  
✔ Multinomial Naive Bayes model for classification  
✔ Accuracy around **95%**  
✔ Interactive UI using Streamlit  
✔ Real-time email prediction

---

## 📂 Project Structure

Email-Spam-Detection/
│
├── dataset/
│ └── emails.csv
│
├── main.ipynb # Training notebook
├── app.py # Streamlit UI
├── spam_model.pkl # Trained model
├── vectorizer.pkl # TF-IDF/Count vectorizer
├── requirements.txt
└── README.md

## ⚙ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Email-Spam-Detection.git
cd Email-Spam-Detection
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Run Application
bash
Copy code
streamlit run app.py
Open in browser: http://localhost:8501
