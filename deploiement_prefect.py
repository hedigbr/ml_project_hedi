"""
Atelier 3 - Étape 05
Déploiement et schédulation des flows Prefect
"""

from pipeline_prefect import all_flow, train_flow, evaluate_flow, all_code_flow

if __name__ == "__main__":

    # Déploiement "all" : exécuté automatiquement chaque jour à 02:00
    all_deployment = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 2 * * *",  # tous les jours à 2h du matin
        tags=["full-pipeline", "mlops"],
        parameters={
            "data_path": "Data/SeoulBikeData.csv",
            "model_path": "Models/rf_tuned_prefect.joblib",
        },
    )

    # Déploiement "train" : disponible pour un lancement manuel
    train_deployment = train_flow.to_deployment(
        name="ml-pipeline-train",
        tags=["training", "mlops"],
        parameters={
            "data_path": "Data/SeoulBikeData.csv",
        },
    )

    # Déploiement "evaluate" : disponible pour un lancement manuel
    evaluate_deployment = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["evaluation", "mlops"],
        parameters={
            "data_path": "Data/SeoulBikeData.csv",
            "model_path": "Models/rf_tuned_prefect.joblib",
        },
    )

    code_deployment = all_code_flow.to_deployment(
        name="ml-pipeline-code",
        tags=["code", "mlops"],
        parameters={},
    )

    # serve() enregistre les déploiements ET reste actif pour exécuter
    # les flows quand ils sont déclenchés (cron ou manuel)
    from prefect import serve

    serve(
        all_deployment,
        train_deployment,
        evaluate_deployment,
        code_deployment
    )