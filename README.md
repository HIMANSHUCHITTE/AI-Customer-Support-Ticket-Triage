# AI Customer Support Ticket Triage

A lightweight, beginner-friendly Machine Learning web application that automatically classifies customer support tickets into **Categories** (`Billing`, `Account`, `Technical`, `Product`) and assigns an operational **Priority** (`Low`, `Medium`, `High`).

Built using **Python**, **Streamlit**, **Pandas**, and **Scikit-learn** with **TF-IDF Vectorization** and **Logistic Regression**.

---

## 📌 Project Overview

In customer support workflows, incoming support requests need to be routed to the appropriate department and prioritized based on urgency. Manual triage is slow and error-prone. This project demonstrates how classic Natural Language Processing (NLP) techniques and supervised machine learning can automate ticket triage with high transparency and zero cloud or API costs.

---

## ✨ Features

- **Automated Multi-Class Classification:**
  - **Category Model:** Classifies tickets into `Billing`, `Account`, `Technical`, or `Product`.
  - **Priority Model:** Predicts urgency levels as `Low`, `Medium`, or `High`.
- **Interpretable NLP Pipeline:** Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to extract textual features and **Logistic Regression** for fast, transparent classification.
- **Interactive Streamlit Web Dashboard:** Clean, responsive UI with one-click test presets, live input area, and dynamic priority badges.
- **Live Model Performance Metrics:** Evaluates both models with stratified five-fold cross-validation and displays mean accuracy.
- **Dataset Inspector:** Built-in interactive view of the underlying support tickets dataset.
- **100% Free & Local:** No external API keys, databases, user logins, or cloud subscriptions required.

---

## 📁 Project Structure

```text
AI_Customer_Support_Triage/
├── app.py              # Main Streamlit web application and ML pipeline
├── tickets.csv         # Labeled dataset of customer support tickets (52 samples)
├── requirements.txt    # Project Python dependencies
├── README.md           # Setup and project documentation
├── PROJECT_REPORT.md   # Academic project report for examination & review
└── screenshots/        # Directory for application screenshots
```

---

## ⚙️ Installation & Setup (Windows)

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed on your system.
- Ensure Python and pip are added to your Windows `PATH`.

### Step 1: Open Windows Command Prompt
Press `Win + R`, type `cmd`, and press **Enter**.

Navigate to the project folder:
```cmd
cd /d "c:\Users\Admin\OneDrive\Desktop\AI_Customer_Support_Triage"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
Run the following command to install `streamlit`, `pandas`, and `scikit-learn`:
```cmd
python -m pip install -r requirements.txt
```

---

## 🚀 Running the Application

To launch the Streamlit web interface, execute:
```cmd
python -m streamlit run app.py
```
*(Or simply `streamlit run app.py` if your Python Scripts folder is in your Windows PATH.)*

Streamlit will start a local development server and open your default browser at:
```text
http://localhost:8501
```

*(If it does not open automatically, copy and paste `http://localhost:8501` into your browser.)*

---

## 🧪 Manual Verification & Sample Test Cases

You can test the trained models using the built-in preset dropdown or by typing the following sample complaints into the text area:

| # | Complaint / Ticket Text | Expected Category | Expected Priority | Reason / Key Signals |
|---|---|---|---|---|
| **1** | *"I was charged twice for my order and need a refund."* | **Billing** | **High** | Words: `charged`, `order`, `refund` trigger payment disputes. |
| **2** | *"I cannot reset my account password."* | **Account** | **High** | Words: `reset`, `account`, `password` signal authentication lockout. |
| **3** | *"The mobile application crashes whenever I open it."* | **Technical** | **High** | Words: `mobile application`, `crashes` indicate critical software failure. |

---

## 🧠 Machine Learning Methodology

1. **Data Preprocessing:** Input ticket text is converted to lowercase and English stop words (e.g., *the*, *is*, *at*) are removed.
2. **Feature Extraction:** `TfidfVectorizer` computes TF-IDF weights, converting text into numerical sparse vectors based on word importance.
3. **Model Training:** Two separate `LogisticRegression` classifiers are trained:
   - Classifier A: Learns text-to-category associations.
   - Classifier B: Learns text-to-priority associations.
4. **Reproducibility:** A fixed random seed (`random_state=42`) is used for cross-validation and model initialization.

---

## 📜 Academic Note for Viva & Review

This project was built to illustrate core data science and software engineering principles in a concise, accessible format for Computer Engineering students. The code prioritizes readability, modular design, and honest reporting of metrics without artificial inflation.
