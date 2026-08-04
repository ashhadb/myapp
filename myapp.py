import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path to dataset (Raw GitHub URL or relative path)
# Replace this URL with your exact raw GitHub link if using remote hosted repo
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at {path}. Check your path or URL.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

@st.cache_resource
def train_and_evaluate_models():
    # Load dataset directly from source
    diabetes_dataset = load_data(DATA_URL)

    X = diabetes_dataset.drop(columns='Outcome')
    Y = diabetes_dataset['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # --- Naive Bayes Model ---
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    Y_pred_nb = nb.predict(X_test)
    
    accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
    precision_nb = precision_score(Y_test, Y_pred_nb, zero_division=0)
    recall_nb = recall_score(Y_test, Y_pred_nb, zero_division=0)
    f1_nb = f1_score(Y_test, Y_pred_nb, zero_division=0)

    # --- KNN Model ---
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred_knn = knn.predict(X_test)
    
    accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
    precision_knn = precision_score(Y_test, Y_pred_knn, zero_division=0)
    recall_knn = recall_score(Y_test, Y_pred_knn, zero_division=0)
    f1_knn = f1_score(Y_test, Y_pred_knn, zero_division=0)

    metrics = {
        'Naive Bayes': {
            'Accuracy': accuracy_nb,
            'Precision': precision_nb,
            'Recall': recall_nb,
            'F1 Score': f1_nb
        },
        'KNN': {
            'Accuracy': accuracy_knn,
            'Precision': precision_knn,
            'Recall': recall_knn,
            'F1 Score': f1_knn
        }
    }
    return metrics

# --- Streamlit UI ---
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title('🩺 Diabetes Prediction Model Comparison')
st.markdown("""
This application compares the performance of **Naive Bayes** and **K-Nearest Neighbors (KNN)**
classification models on the diabetes dataset. Select a model from the dropdown to see its evaluation metrics.
""")

metrics_data = train_and_evaluate_models()

# Dropdown menu for model selection
selected_model = st.selectbox(
    'Select a Classification Model:',
    ('Naive Bayes', 'KNN')
)

st.write(f"### Metrics for {selected_model}:")

if selected_model:
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Accuracy", value=f"{metrics_data[selected_model]['Accuracy']:.3f}")
        st.metric(label="Precision", value=f"{metrics_data[selected_model]['Precision']:.3f}")
    with col2:
        st.metric(label="Recall", value=f"{metrics_data[selected_model]['Recall']:.3f}")
        st.metric(label="F1 Score", value=f"{metrics_data[selected_model]['F1 Score']:.3f}")
