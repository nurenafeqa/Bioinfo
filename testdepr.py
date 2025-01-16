import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Helper functions
def preprocess_data(df, normalize=True):
    """Preprocess dataset: normalize numeric columns and handle date columns."""
    # Convert date columns to datetime and extract features
    date_columns = df.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col + '_year'] = df[col].dt.year
        df[col + '_month'] = df[col].dt.month
        df[col + '_day'] = df[col].dt.day
        df[col + '_hour'] = df[col].dt.hour
        df = df.drop(columns=[col])  # Drop original date column

    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])

    if normalize:
        scaler = StandardScaler()
        numeric_cols = pd.DataFrame(
            scaler.fit_transform(numeric_cols),
            columns=numeric_cols.columns
        )
        df[numeric_cols.columns] = numeric_cols

    # Encode categorical columns using LabelEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df


def train_model(df, target, model_type):
    """Train Random Forest or Gradient Boosting model."""
    X = df.drop(columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        return None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, cm, report


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Streamlit UI
def main():
    st.title("Depression and Anxiety Detection App")

    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Choose Mode", ["Train Model", "Use Model"])

    if tab == "Train Model":
        st.header("Train a Classification Model")
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:", df.head())

            if st.checkbox("Normalize Dataset (Z-score)"):
                df = preprocess_data(df)
                st.write("Normalized Data:", df.head())

            model_type = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])
            target_column = st.selectbox("Select Target Column", df.columns)

            if st.button("Train Model"):
                model, cm, report = train_model(df, target_column, model_type)
                st.write(f"{model_type} Confusion Matrix:", cm)
                st.write(f"{model_type} Classification Report:", report)
                save_model(model, f"{model_type.lower().replace(' ', '_')}_model.pkl")

    elif tab == "Use Model":
        st.header("Use Pretrained Model for Prediction")

        uploaded_file = st.file_uploader("Upload Dataset for Prediction", type=["csv"])
        model_file = st.file_uploader("Upload Pretrained Model", type=["pkl"])

        if uploaded_file and model_file:
            df = pd.read_csv(uploaded_file)
            model = load_model(model_file)
            df = preprocess_data(df, normalize=False)  # Apply preprocessing to the input data
            predictions = model.predict(df)
            st.write("Predictions:", predictions)

if __name__ == "__main__":
    main()
