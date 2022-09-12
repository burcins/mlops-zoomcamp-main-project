#!/usr/bin/env python
# coding: utf-8



from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow


def run():

    #client = MlflowClient()

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    

    # select the model with the highest accuracy
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs( 
    experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.accuracy_score DESC"]
    )[0]

    # register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    print("Accuracy score of the best model" , best_run.data.metrics['accuracy_score'])
    model_name = "wine_quality_clf" 
    mlflow.register_model(model_uri=model_uri, name=model_name)
    
    return model_uri, model_name 


def register_model(model_name):

    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Staging",
        archive_existing_versions=True
    )

if __name__ == '__main__':
    EXPERIMENT_NAME = 'wine_quality_clf'
    mlflow.set_tracking_uri("sqlite:///mlflow-experiments.db")
    client = MlflowClient()
    model_uri, model_name  = run()
    register_model(model_name)
    
    latest_versions = client.get_latest_versions(name=model_name)

    for version in latest_versions:
        print(f"version: {version.version}, stage: {version.current_stage}")
    
