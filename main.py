#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

import optuna
from sklearn.pipeline import Pipeline
import joblib

import mlflow
from mlflow.tracking import MlflowClient

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import RRuleSchedule

#from prefect.flow_runners import SubprocessFlowRunner
from prefect.infrastructure import Process

from datetime import timedelta


import warnings
warnings.filterwarnings('ignore')



@task
def prepare_data():
    

    redwine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    redwine['type'] = 'red'

    whitewine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
    whitewine['type'] = 'white'

    df = pd.concat([redwine, whitewine], axis = 0).reset_index(drop=True)
    
    numeric_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

    df['quality'] =  np.where(df.quality.isin([3,4]), '3-4', np.where(df.quality.isin([8,9]), '8-9', df.quality))
    df['quality'] =  df.quality.astype('object')

    return df, numeric_columns


@task
def train_model(df, numeric_columns):
    
    with mlflow.start_run():

        date = datetime.datetime.today().strftime("%Y-%m-%d/%H_%M")
        X_train, X_test, y_train, y_test = train_test_split(df.drop('quality', axis=1), df.quality,
                                                        stratify=df.quality, 
                                                        test_size=0.3,
                                                        random_state=123)


        pipe = Pipeline([
            ('column_transformer', ColumnTransformer([
                ('one-hot', OneHotEncoder(handle_unknown='ignore'), ['type']),
                ('scaler', StandardScaler(), numeric_columns)], remainder='drop')),
            ('model', OneVsRestClassifier(xgb.XGBClassifier(random_state= 123, eval_metric='logloss')))])
            
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        #joblib.dump(pipe, 'pipeline-{}.pkl'.format(date))
                        
        mlflow.set_tag("developer", "burcin")

        mlflow.sklearn.log_model(pipe, artifact_path="models/model-{}".format(date))
            
    return X_train, X_test, y_train, y_test, pipe





def xgb_objective(trial, X_train, y_train, X_test, y_test, numeric_columns):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")

        params = {
            "min_split_loss": trial.suggest_int("min_split_loss", 0, 10, step=1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.5, 1, step=0.1),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "objective": trial.suggest_categorical(
                "objective", ["binary:logistic", "binary:hinge"]
            ),
            "eval_metric" : "logloss",
            "random_state": 123

        }
        # Perform CV
        mlflow.log_params(params)   
            
        model = OneVsRestClassifier(xgb.XGBClassifier(**params))
        
        
        pipe = Pipeline([
            ('column_transformer', ColumnTransformer([
                ('one-hot', OneHotEncoder(handle_unknown='ignore'), ['type']),
                ('scaler', StandardScaler(), numeric_columns)], remainder='drop')),
            ('model', model)])
            
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        score = accuracy_score(y_test, preds)
        
        mlflow.log_metric("accuracy_score", score)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
    
        
    return score
        
@task       
def run_experiments(X_train, y_train, X_test, y_test, numeric_columns):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create study that maximize
    study = optuna.create_study(direction="maximize")

    func = lambda trial: xgb_objective(trial,X_train, y_train, X_test, y_test, numeric_columns)

    # Start optimizing with 10 trials
    study.optimize(func, n_trials=10)
    
    return print(study.best_value)
        


#@flow(task_runner=SequentialTaskRunner())
@flow
def main(date="2021-09-12"):
    mlflow.set_tracking_uri("sqlite:///mlflow-experiments.db")
    mlflow.set_experiment("wine_quality_clf")    
    
    df, numeric_columns = prepare_data()
    
    X_train, X_test, y_train, y_test, pipe = train_model(df, numeric_columns)
        
    run_experiments(X_train, y_train, X_test, y_test, numeric_columns)
   

    
if __name__ == '__main__':
    main()


