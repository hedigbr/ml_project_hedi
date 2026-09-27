import subprocess
import argparse
from prefect import task, flow
from Models.model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

TARGET_FILES = ["Models/model_pipeline.py", "Models/main.py", "pipeline_prefect.py"]
TEST_FILES = ["tests/test_model_pipeline.py"]


@task(name="Installer les dépendances")
def install_dependencies():
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)


@task(name="Formatage du code")
def format_code():
    subprocess.run(["black"] + TARGET_FILES, check=True)


@task(name="Qualité du code")
def code_qualite():
    subprocess.run(["pylint"] + TARGET_FILES, check=False)


@task(name="Sécurité du code")
def code_security():
    subprocess.run(["bandit"] + TARGET_FILES, check=False)


@task(name="Test Unitaire du code")
def code_UnitTests():
    subprocess.run(["pytest"] + TEST_FILES, check=True)


@task(name="Préparation des données")
def prepare_data_task(data_path):

    X_train, X_test, y_train, y_test, scaler, pca = prepare_data(data_path)

    return X_train, X_test, y_train, y_test


@task(name="Entraînement du modèle")
def train_model_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Sauvegarder le modèle")
def save_model_task(model, model_path):
    save_model(model, model_path)


@task(name="Charger le modèle")
def load_model_task(model_path):
    return load_model(model_path)


@task(name="Évaluation du modèle")
def evaluate_model_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


@flow(name="train")
def train_flow(data_path):

    X_train, X_test, y_train, y_test = prepare_data_task(data_path)

    model = train_model_task(X_train, y_train)

    return model


@flow(name="evaluate")
def evaluate_flow(data_path, model_path):

    # On prépare les données pour récupérer X_test et y_test
    X_train, X_test, y_train, y_test = prepare_data_task(data_path)

    model = load_model_task(model_path)

    metrics = evaluate_model_task(model, X_test, y_test)

    return metrics


@flow(name="all")
def all_flow(data_path, model_path):

    X_train, X_test, y_train, y_test = prepare_data_task(data_path)

    model = train_model_task(X_train, y_train)

    save_model_task(model, model_path)

    metrics = evaluate_model_task(model, X_test, y_test)

    return metrics


@flow(name="code")
def all_code_flow():
    format_code()
    code_qualite()
    code_security()
    code_UnitTests()


def pipeline_prefect():

    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")

    parser.add_argument(
        "--flow", choices=["all", "entrainement", "evaluate", "code"], required=True
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow("Data/SeoulBikeData.csv", "Models/rf_tuned_prefect.joblib")

    elif args.flow == "entrainement":
        train_flow("Data/SeoulBikeData.csv")

    elif args.flow == "evaluate":
        evaluate_flow("Data/SeoulBikeData.csv", "Models/rf_tuned_prefect.joblib")

    elif args.flow == "code":
        all_code_flow()


if __name__ == "__main__":
    pipeline_prefect()
