
import streamlit as st
import numpy as np
import pandas as pd
from google.colab import drive # For mounting Google Drive in Colab
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Mount Google Drive (this will prompt for authentication if not already mounted by this process)
@st.cache_resource
def mount_drive():
    try:
        drive.mount('/content/drive', force_remount=True)
        return True
    except Exception as e:
        st.error(f"Could not mount Google Drive: {e}. Please ensure you're running in Colab and grant permissions.")
        return False

if mount_drive():
    file_path = '/content/drive/MyDrive/Datasets/diabetes.csv'
else:
    st.stop() # Stop the app if drive cannot be mounted

@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at {path}. Please check the path and ensure your Google Drive is mounted correctly.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

@st.cache_resource
def train_and_evaluate_models():
    diabetes_dataset = load_data(file_path)

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
    st.metric(label="Accuracy", value=f"{metrics_data[selected_model]['Accuracy']:.3f}")
    st.metric(label="Precision", value=f"{metrics_data[selected_model]['Precision']:.3f}")
    st.metric(label="Recall", value=f"{metrics_data[selected_model]['Recall']:.3f}")
    st.metric(label="F1 Score", value=f"{metrics_data[selected_model]['F1 Score']:.3f}")