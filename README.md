# 🗞️ BERT-Powered News Classifier

## 📌 Project Overview
I built a custom NLP classifier using **BERT (Bidirectional Encoder Representations from Transformers)**. It's fine-tuned on the AG News dataset to categorize headlines into four distinct sectors with high precision.

### 📊 Performance
- **Model:** `bert-base-uncased`
- **Accuracy:** 86.8% (on a 2,000 sample test run)
- **Frameworks:** PyTorch, Hugging Face Transformers, Streamlit

## ⚙️ Tech Stack
- **Python 3.14**
- **Hugging Face `Trainer` API** for fine-tuning.
- **Streamlit** for the frontend UI.
- **Git LFS** (or Hugging Face Hub) for model hosting.

## 🚀 How to Run Locally
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/News-Classifier-BERT.git](https://github.com/YOUR_USERNAME/News-Classifier-BERT.git)
   cd News-Classifier-BERT