# Academic Project Report

## Project Title: AI Customer Support Ticket Triage System

**Course:** Computer Engineering Mini-Project / Natural Language Processing  
**Student Level:** Third-Year B.Tech / B.E. Computer Engineering  
**Academic Year:** 2025–2026  

---

### 1. Abstract
Customer support operations in modern digital enterprises encounter large volumes of incoming user queries, bug reports, and grievances daily. Manual sorting, routing, and prioritization of tickets introduce bottlenecks, slow down response times, and increase the risk of delayed resolutions for critical business failures. 

This project presents an intelligent, lightweight **AI Customer Support Ticket Triage** system designed to automatically classify customer support tickets into distinct functional departments (**Billing**, **Account**, **Technical**, and **Product**) while concurrently determining operational severity (**Low**, **Medium**, and **High**). Developed with **Python**, **Pandas**, **Scikit-learn**, and **Streamlit**, the system employs **Term Frequency-Inverse Document Frequency (TF-IDF)** feature extraction coupled with two supervised **Logistic Regression** classifiers. The resulting web application operates entirely on local hardware with zero external cloud dependencies, minimal latency, and transparent model interpretability.

---

### 2. Problem Statement
Customer support desks typically receive hundreds of heterogeneous tickets every hour. The core operational challenges include:
1. **Routing Inefficiencies:** Manual ticket assignment to appropriate technical or customer support personnel is repetitive and prone to human misclassification.
2. **Delayed Emergency Response:** High-urgency issues (e.g., unauthorized transactions, server crashes) often sit in queues behind minor informational inquiries.
3. **High Architectural Complexity of Modern LLMs:** While large language models (LLMs) can classify text, deploying cloud-based deep learning APIs introduces prohibitive costs, latency, privacy compliance challenges, and reliance on external network access.

Hence, there is a clear demand for a simple, fast, transparent, and self-contained machine learning solution capable of performing triage reliably without complex infrastructure.

---

### 3. Objectives
The key objectives of this project are:
- To construct a realistic, balanced dataset of customer support tickets across four business categories and three priority levels.
- To implement an automated Natural Language Processing (NLP) pipeline utilizing **TF-IDF Vectorization** for numerical feature extraction.
- To train two independent **Logistic Regression** multi-class models for category classification and priority assessment.
- To achieve reproducible model evaluation through fixed-seed, stratified five-fold cross-validation.
- To build an intuitive, interactive, and responsive web dashboard using **Streamlit** for real-time ticket triage and model introspection.
- To maintain clean, modular, and beginner-friendly code architecture suitable for third-year engineering review and viva examination.

---

### 4. Technologies Used

| Technology / Tool | Version / Library | Purpose in Project |
|---|---|---|
| **Python** | 3.11+ | Primary programming language |
| **Streamlit** | >= 1.30.0 | Lightweight front-end web dashboard and reactive interface |
| **Pandas** | >= 2.0.0 | Tabular dataset management, CSV ingestion, and data structuring |
| **Scikit-learn** | >= 1.3.0 | Text vectorization (`TfidfVectorizer`), model training (`LogisticRegression`), and evaluation metrics |
| **TF-IDF Vectorizer** | Scikit-learn Module | Converts raw textual ticket strings into weighted numerical feature vectors |
| **Logistic Regression** | Scikit-learn Linear Model | Multi-class linear classification for category and priority |

---

### 5. Methodology

The machine learning methodology adheres to the standard supervised learning workflow:

#### 5.1 Dataset Design
A balanced dataset (`tickets.csv`) consisting of 52 realistic customer support tickets was compiled.
- **Features:** `ticket_text` (raw textual description of the user issue).
- **Target Variable 1 (Category):** `Billing`, `Account`, `Technical`, `Product` (13 samples per category).
- **Target Variable 2 (Priority):** `Low`, `Medium`, `High` (balanced across urgency types).

#### 5.2 Text Preprocessing and Vectorization
Raw customer inputs contain varying lengths, cases, and punctuation. The NLP stage applies:
1. **Case Normalization:** Converts all tokens to lowercase to prevent vocabulary duplication.
2. **Stop Word Removal:** Filters standard English stop words (e.g., *the*, *is*, *in*, *and*) that carry minimal domain-specific discriminatory value.
3. **TF-IDF Calculation:** Evaluates word importance within a ticket relative to the entire corpus using the formula:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
   Where:
   - $\text{TF}(t, d)$ is the frequency of term $t$ in ticket $d$.
   - $\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$.

#### 5.3 Supervised Model Training
Two separate multi-class Logistic Regression models are trained using Scikit-learn:
- **Model A (Category Classifier):** Optimizes multi-class cross-entropy loss over the four department labels.
- **Model B (Priority Classifier):** Optimizes multi-class cross-entropy loss over the three priority tiers.
- **Regularization & Optimization:** Uses L2 regularization with L-BFGS solver (`max_iter=1000`) and a fixed random seed (`random_state=42`) to guarantee identical results across multiple runs.
- **Evaluation:** Uses stratified five-fold cross-validation. Each fold fits its own TF-IDF vectorizer to avoid data leakage; after evaluation, the live classifiers are refit on all labeled tickets.

---

### 6. System Workflow

```
+-----------------------------------------------------------+
|               Incoming Customer Support Ticket            |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
|             Data Cleaning & Text Preprocessing            |
|       (Lowercasing, English Stop Words Removal)           |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
|              TF-IDF Feature Representation                |
|             (Transformed Sparse Vector Space)             |
+-----------------------------------------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
+----------------------------+  +----------------------------+
|  Model A: Category Model   |  |  Model B: Priority Model   |
|   (Logistic Regression)    |  |   (Logistic Regression)    |
+----------------------------+  +----------------------------+
              |                               |
              v                               v
    Predicted Department              Predicted Urgency
  [Billing / Account / Tech]        [High / Medium / Low]
              |                               |
              +---------------+---------------+
                              |
                              v
+-----------------------------------------------------------+
|          Interactive Streamlit Web Dashboard Output       |
|    - Category Badge       - Priority Status Indicator     |
+-----------------------------------------------------------+
```

---

### 7. Testing & Verification

The system was verified against a set of functional test cases spanning multiple customer support domains:

| Test ID | Input Ticket Text | Expected Category | Expected Priority | Verified Match (Run Results) |
|---|---|---|---|---|
| **TC-01** | *"I was charged twice for my order and need a refund."* | Billing | High | **Category: Billing, Priority: High** (Exact Match) |
| **TC-02** | *"I cannot reset my account password."* | Account | High | **Category: Account, Priority: Medium** (Category Match, Priority shifted due to small training split) |
| **TC-03** | *"The mobile application crashes whenever I open it."* | Technical | High | **Category: Technical, Priority: High** (Exact Match) |
| **TC-04** | *"Can you please send me the invoice for my purchase last month?"* | Billing | Low | **Category: Billing, Priority: High** (Category Match) |
| **TC-05** | *"The product arrived with defective parts and does not turn on."* | Product | High | **Category: Product, Priority: High** (Exact Match) |

#### Verification Procedure
1. Launch the application using `python -m streamlit run app.py` (or `streamlit run app.py`).
2. Observe the mean five-fold cross-validation accuracy reported in the top metric cards.
3. Select each test case from the quick dropdown or paste the text directly into the text area.
4. Click **Predict Category & Priority** and record the displayed output.

---

### 8. Results & Limitations

### 8.1 Empirical Results (Five-Fold Cross-Validation)
- **Total Dataset Size:** 52 tickets, evaluated with stratified five-fold cross-validation (`random_state=42`)
- **Category Model Mean Accuracy:** **73.3%**
- **Priority Model Mean Accuracy:** **37.1%**

#### 8.2 Analysis of Results & Limitations
1. **Topical vs. Urgency Signals:** The Category Model achieved higher mean accuracy (**73.3%**) because domain vocabulary (e.g., *refund, charged, password, crashes, keyboard*) clusters around departments. Priority is harder to infer from short text because urgency words are sparse, cross-departmental, and context-dependent; its mean accuracy was **37.1%**.
2. **Small-Data Effect:** Each validation fold contains only 10 or 11 tickets, so individual predictions have a large effect on fold scores. The five-fold mean is more representative than one small holdout, but 52 examples remain insufficient for a reliable production estimate.
3. **Out-of-Vocabulary (OOV) Tokens:** Pure TF-IDF creates a fixed vocabulary from training data and cannot infer synonyms or unseen words without contextual embeddings.
4. **Independent Classifiers:** Category and Priority are modeled as two separate single-target classifiers rather than a joint multi-task architecture.
5. **Context Window:** The system processes short single-turn customer complaints rather than multi-turn conversational email threads or file attachments.

---

### 9. Conclusion
The **AI Customer Support Ticket Triage** project successfully demonstrates how foundational Machine Learning techniques—specifically TF-IDF vectorization and Logistic Regression—can address practical enterprise routing problems efficiently. The solution is straightforward, computationally efficient, runs locally without external dependencies, and provides an accessible, fully explainable baseline for Computer Engineering students.
