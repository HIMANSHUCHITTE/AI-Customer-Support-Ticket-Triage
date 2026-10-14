import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Support Ticket Triage",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    [data-testid="stAppDeployButton"] { display: none !important; }
    h1 { color: var(--text-color); font-size: 2rem !important; }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.24);
        border-radius: 6px;
        padding: 14px 16px;
    }
    [data-testid="stBaseButton-primary"] {
        background: #167c70;
        border-color: #167c70;
        color: #ffffff;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background: #11645a;
        border-color: #11645a;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Model Training and Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_data_and_train(csv_filename="tickets.csv"):
    """
    Evaluates category and priority models, then fits both on the full dataset.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_filename)

    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}. Please make sure {csv_filename} exists.")
        st.stop()

    # Load dataset
    df = pd.read_csv(csv_path)

    # Feature and targets
    X = df["ticket_text"]
    y_category = df["category"]
    y_priority = df["priority"]

    # Evaluate with stratified folds; each pipeline fits its vectorizer per fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    category_pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english", lowercase=True),
        LogisticRegression(random_state=42, max_iter=1000)
    )
    priority_pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english", lowercase=True),
        LogisticRegression(random_state=42, max_iter=1000)
    )
    category_accuracy = cross_val_score(
        category_pipeline, X, y_category, cv=cv, scoring="accuracy"
    ).mean()
    priority_accuracy = cross_val_score(
        priority_pipeline, X, y_priority, cv=cv, scoring="accuracy"
    ).mean()

    # Fit the live classifiers on every labeled ticket after evaluation.
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X_tfidf = vectorizer.fit_transform(X)
    category_model = LogisticRegression(random_state=42, max_iter=1000)
    category_model.fit(X_tfidf, y_category)
    priority_model = LogisticRegression(random_state=42, max_iter=1000)
    priority_model.fit(X_tfidf, y_priority)

    return (
        df,
        vectorizer,
        category_model,
        priority_model,
        category_accuracy,
        priority_accuracy,
        cv.get_n_splits()
    )

# Load data and models
df, vectorizer, category_model, priority_model, cat_acc, pri_acc, cv_folds = load_data_and_train()

# -----------------------------------------------------------------------------
# User Interface Header
# -----------------------------------------------------------------------------
st.title("AI Customer Support Ticket Triage")
st.markdown("Route incoming customer tickets by **category** and **priority**.")

st.divider()

# -----------------------------------------------------------------------------
# Model Performance Metrics
# -----------------------------------------------------------------------------
st.subheader("Model performance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tickets", f"{len(df)}")
with col2:
    st.metric("Cross-validation", f"{cv_folds} folds")
with col3:
    st.metric("Category Accuracy", f"{cat_acc * 100:.1f}%")
with col4:
    st.metric("Priority Accuracy", f"{pri_acc * 100:.1f}%")

st.caption(
    f"Scores are mean stratified {cv_folds}-fold cross-validation accuracy on {len(df)} tickets. "
    "Results may vary with a larger, more diverse dataset."
)

st.divider()

# -----------------------------------------------------------------------------
# Ticket Classification Section
# -----------------------------------------------------------------------------
st.subheader("Classify a ticket")

# Quick test prompt selector to make live testing effortless
preset_options = [
    "-- Select an example or write your own below --",
    "I was charged twice for my order and need a refund.",
    "I cannot reset my account password.",
    "The mobile application crashes whenever I open it.",
    "Can you please send me the invoice for my purchase last month?",
    "The product arrived with defective parts and does not turn on."
]
selected_preset = st.selectbox("Example tickets", preset_options)

default_text = ""
if selected_preset != preset_options[0]:
    default_text = selected_preset

user_ticket = st.text_area(
    "Ticket text",
    value=default_text,
    height=120,
    placeholder="Describe the customer's issue..."
)

predict_clicked = st.button("Classify ticket", type="primary")

if predict_clicked:
    cleaned_text = user_ticket.strip()
    if not cleaned_text:
        st.warning("Enter ticket text to continue.")
    else:
        # Transform input using fitted TF-IDF Vectorizer
        input_tfidf = vectorizer.transform([cleaned_text])

        # Predict Category and Priority
        predicted_category = category_model.predict(input_tfidf)[0]
        predicted_priority = priority_model.predict(input_tfidf)[0]

        # Display Results
        st.success("Triage complete")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(
                f"""
                <div style="background-color: #f0f4f8; padding: 18px; border-radius: 8px; border-left: 5px solid #2b6cb0;">
                    <h4 style="margin: 0; color: #2b6cb0;">Predicted Category</h4>
                    <p style="font-size: 24px; font-weight: bold; margin: 8px 0 0 0; color: #1a202c;">
                        {predicted_category}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Color highlight according to priority
        priority_colors = {
            "High": ("#fed7d7", "#9b2c2c", "#c53030"),
            "Medium": ("#feebc8", "#7b341e", "#dd6b20"),
            "Low": ("#c6f6d5", "#22543d", "#38a169")
        }
        bg_col, txt_col, border_col = priority_colors.get(
            predicted_priority, ("#edf2f7", "#2d3748", "#4a5568")
        )

        with res_col2:
            st.markdown(
                f"""
                <div style="background-color: {bg_col}; padding: 18px; border-radius: 8px; border-left: 5px solid {border_col};">
                    <h4 style="margin: 0; color: {txt_col};">Predicted Priority</h4>
                    <p style="font-size: 24px; font-weight: bold; margin: 8px 0 0 0; color: {txt_col};">
                        {predicted_priority}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()

# -----------------------------------------------------------------------------
# Sample Tickets Explorer
# -----------------------------------------------------------------------------
with st.expander("View sample tickets"):
    st.markdown("Customer support tickets used to train the model:")
    st.dataframe(df, use_container_width=True, height=280)
    st.caption(f"Showing all {len(df)} records across 4 categories and 3 priorities.")
