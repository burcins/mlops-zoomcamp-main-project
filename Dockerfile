FROM python:3.9.7-slim

RUN apt-get update -y

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy 

COPY ["main.py" , "model_registry.py", "./"]

ENTRYPOINT pipenv shell 

EXPOSE 5000

EXPOSE 4200

RUN python3 main.py

RUN python3 model_registry.py