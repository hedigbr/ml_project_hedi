"""
Atelier 2 - Modularisation du Code
Projet : Seoul Bike Sharing Demand
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(file_path):
    """Charge et prétraite les données, puis prépare train/test."""
    df = pd.read_csv(file_path, encoding="latin1")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.fillna(0).astype(int)
        df = df.drop(columns=["Date"])

    target = "Rented Bike Count"
    if target not in df.columns:
        raise ValueError(f"La colonne cible '{target}' est absente.")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    y = df[target]
    X = df.drop(columns=[target])

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_probe = PCA()
    pca_probe.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_probe.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative_variance >= 0.95) + 1)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.20, random_state=42
    )

    print(f"Variables après PCA : {n_components}")
    print(f"Variance expliquée : {pca.explained_variance_ratio_.sum():.2%}")

    return X_train, X_test, y_train, y_test, scaler, pca


def train_model(X_train, y_train):
    """Entraîne un Random Forest de manière classique."""

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Modèle Random Forest entraîné avec succès.")

    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle avec MSE, RMSE et R²."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {"mse": mse, "rmse": rmse, "r2": r2}


def save_model(model, output_path):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, output_path)
    print(f"Modèle sauvegardé : {output_path}")


def load_model(model_path):
    """Charge un modèle sauvegardé."""
    model = joblib.load(model_path)
    print(f"Modèle chargé : {model_path}")
    return model
