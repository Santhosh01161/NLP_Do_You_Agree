# 🤖 NLP: Do You Agree? (Sentence-BERT & NLI) 🚀

Welcome to the **NLP_Do_You_Agree** repository! This project demonstrates the end-to-end process of building a **BERT** model from scratch, fine-tuning it into a Siamese **Sentence-BERT (SBERT)** architecture for Natural Language Inference (NLI), and deploying it as a professional web application.

<img width="875" height="784" alt="Screenshot 2026-02-15 at 1 05 41 am" src="https://github.com/user-attachments/assets/12f3a95c-fb96-486e-aa00-da462b52f74a" />

---

## 📁 Project Structure

* **`app/`**: Contains the Flask web application, tokenizer assets, and inference logic. 🌐
* **`A4_Do_You_Agree.ipynb`**: The main Jupyter Notebook containing model training, evaluation, and Task 1-3. 📓
* **`NLP_A4 Do You Agree.pdf`**: GitHub Repo and Task 3 Answer. 📄
* **`.gitignore`**: Configuration to keep the repository clean from large model weights. 🛡️

---

## 🛠️ Tasks Overview

### Task 1: Training BERT from Scratch 🏗️

Implemented a **Bidirectional Encoder Representations from Transformers (BERT)** model using:
* **Masked Language Modeling (MLM)** objective.
* **Dataset**: `rojagtap/bookcorpus` (subset of 150k samples).
* **Architecture**: Multi-head attention, GELU activation, and custom Layer Normalization.

### Task 2: Sentence-BERT (Siamese Network) 👯
Adapted the BERT backbone into a Siamese architecture to handle sentence pairs:
* **Dataset**: Stanford Natural Language Inference (SNLI).
* **Objective**: Classify sentence relationships into *Entailment*, *Neutral*, or *Contradiction*.
* **Pooling**: Implemented Mean Pooling to derive fixed-size sentence embeddings (u and v).

### Task 3: Evaluation & Analysis 📊
Calculated performance metrics using `classification_report`.
* **Accuracy achieved**: ~41% (on a 800-sample computational subset).
* **Key Discussion**: Analyzed VRAM constraints, the impact of small-batch training, and gradient accumulation strategies.

### Task 4: Web Application 💻
A Flask-based web interface where users can input two sentences to check their semantic relationship and similarity score in real-time.

---

#### **📸 Prediction Results & Similarity Score**
<img width="875" height="651" alt="Screenshot 2026-02-15 at 1 02 39 am" src="https://github.com/user-attachments/assets/14ba8b78-1f1a-4c6d-ba29-1ef3d40e7005" />

<img width="875" height="651" alt="Screenshot 2026-02-15 at 1 02 17 am" src="https://github.com/user-attachments/assets/dc79a097-cb95-44b3-bc6d-3fab6c3b2c7a" />

<img width="875" height="651" alt="Screenshot 2026-02-15 at 1 04 57 am" src="https://github.com/user-attachments/assets/f3eb425a-4adc-4199-942e-d621f9f793d0" />

---

## 🚀 How to Run the App

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Santhosh01161/NLP_Do_You_Agree.git](https://github.com/Santhosh01161/NLP_Do_You_Agree.git)
   cd NLP_Do_You_Agree/app
