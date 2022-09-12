# END TO END MLOPS PROJECT
## MULTICLASS CLASSIFICATION MODEL DEVELOPMENT AND ORCHESTRATION OPERATIONS 

Problem

This is a simple end-to-end mlops project organized by [MLOpsZoomCamp](https://github.com/DataTalksClub/mlops-zoomcamp) 

### DATA 
Citation Request:
  This dataset is public available for research. The details are described in [Cortez et al., 2009]. 

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib
           

### PROBLEM

This is a simple end-to-end mlops project including data exploration, experimenting and model development as well as experiment tracking with MLFlow and ochestration with Prefect as workflow tool to deploying the model as a web service.

For this project publicly available Wine data and was used and a simple multiclass classification model was developed to predict wine quality and assign a quality rate between 3 to 9, based on ingredients of the product as predictors. 


## PROJECT SETUP

### AUTO MODEL REGISTRATION

Clone the project from the repository

      git clone https://github.com/burcins/mlops-zoomcamp-main-project/

Change to mlops-project directory

      cd mlops-zoomcamp-main-project

Setup and install project dependencies

      make setup
      
![Done](https://github.com/burcins/mlops-zoomcamp-main-project/blob/master/screenshots/dockerdone.png?raw=true)

Start Local Prefect Server

In a new terminal window or tab run the command below to start prefect orion server

      prefect orion start

![Prefect](https://github.com/burcins/mlops-zoomcamp-main-project/blob/master/screenshots/prefect.png?raw=true)

In a new terminal window or tab run the command below to start Local Mlflow Server

      mlflow ui --backend-store-uri sqlite:///mlflow-experiments.db
      
![MLFlow](https://github.com/burcins/mlops-zoomcamp-main-project/blob/master/screenshots/mlflow.png?raw=true)

### MANUALLY TRACK EXPERIMENTS

Above process automatically run experiments and picked the best model by accuracy and register that model in "Staging" stage. 
If one prefer to run manual experiments one should install environment with this command ;

      pipenv install 
      
Then manually trigger main.py file with this command;

      python3 main.py

This will run experiments and log the records to both MLFlow experiment page as well as create run in Prefect. 


## Model as a Webservice

Webservice itself allows user to run apart from the main tasks. 

After changing directory to webservice folder you can simply run docker to build a docker container, and then when you run the container it will test the model to predict a specific wine's quality based on its features sent as a JSON file.

      sudo docker build -t wine-quality-prediction-service:v1 .
      
      sudo docker run -it --rm -p 9696:9696 wine-quality-prediction-service:v1
      
  

