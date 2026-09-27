"""
Atelier 2 - Modularisation du Code
Point d'entrée du pipeline ML.
"""

import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Seoul Bike Sharing")
    parser.add_argument(
        "--data",
        required=True,
        help="Chemin vers SeoulBikeData.csv",
    )
    parser.add_argument(
        "--model",
        default="Models/rf_tuned.joblib",
        help="Chemin de sauvegarde du modèle",
    )
    args = parser.parse_args()

    print("\n=== 1. PREPARATION DES DONNEES ===")
    X_train, X_test, y_train, y_test, scaler, pca = prepare_data(args.data)

    print("\n=== 2. ENTRAINEMENT ===")
    model = train_model(X_train, y_train)

    print("\n=== 3. EVALUATION ===")
    metrics = evaluate_model(model, X_test, y_test)

    print("\n=== 4. SAUVEGARDE ===")
    save_model(model, args.model)

    print("\n=== 5. TEST DU CHARGEMENT ===")
    load_model(args.model)

    print("\nPipeline terminé avec succès.")
    print(metrics)


if __name__ == "__main__":
    main()
