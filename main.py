#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

import optuna
from sklearn.metrics import cohen_kappa_score
import pickle

import mlflow
from mlflow.tracking import MlflowClient

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import RRuleSchedule

from prefect.flow_runners import SubprocessFlowRunner
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
def train_model(df, numeric_columns, date):
    with mlflow.start_run():


        X_train, X_test, y_train, y_test = train_test_split(df.drop('quality', axis=1), df.quality,
                                                        stratify=df.quality, 
                                                        test_size=0.3,
                                                        random_state=123)


        ct = ColumnTransformer([
        ('one-hot', OneHotEncoder(handle_unknown='ignore'), ['type']),
        ('scaler', StandardScaler(), numeric_columns)], remainder='drop')


        model = OneVsRestClassifier(xgb.XGBClassifier(random_state= 123, eval_metric='logloss'))
        
        X_train_transformed = ct.fit_transform(X_train)
        X_test_transformed = ct.transform(X_test)
        
        model.fit(X_train_transformed, y_train)
        preds = model.predict(X_test_transformed)


        with open('xgb.bin', 'wb') as f_out:
            pickle.dump((ct, ), f_out)
                        
            mlflow.set_tag("developer", "burcin")

        mlflow.sklearn.log_model(model, artifact_path="models/model-{}".format(date))
            
    return X_train_transformed, X_test_transformed, y_train, y_test, model, ct





def xgb_objective(trial, X_train, y_train, X_test, y_test):
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
        
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        score = accuracy_score(y_test, preds)
        
        mlflow.log_metric("accuracy_score", score)
        
    return score
        
@task       
def run_experiments(X_train_transformed, y_train, X_test_transformed, y_test):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create study that maximize
    study = optuna.create_study(direction="maximize")

    func = lambda trial: xgb_objective(trial,X_train_transformed, y_train, X_test_transformed, y_test)

    # Start optimizing with 10 trials
    study.optimize(func, n_trials=10)
    
    return print(study.best_value)
        


@flow(task_runner=SequentialTaskRunner())
def main(date="2021-09-12"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wine_qualit_clf")    
    
    df, numeric_columns = prepare_data()
    
    X_train_transformed, X_test_transformed, y_train, y_test, model, ct = train_model(df, numeric_columns, date)
        
    run_experiments(X_train_transformed, y_train, X_test_transformed, y_test)
   

    
if __name__ == '__main__':
    main()


