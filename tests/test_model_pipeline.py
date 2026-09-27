"""Tests unitaires pour model_pipeline.py"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Models.model_pipeline import train_model, evaluate_model


def test_train_model_returns_fitted_model():
    X_train = np.random.rand(20, 5)
    y_train = np.random.rand(20)

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, "predict")


def test_evaluate_model_returns_metrics():
    X_train = np.random.rand(20, 5)
    y_train = np.random.rand(20)
    X_test = np.random.rand(5, 5)
    y_test = np.random.rand(5)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics