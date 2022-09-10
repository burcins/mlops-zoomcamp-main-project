FROM python:3.8.12-slim

RUN apt-get update -y

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy 

COPY ["main.py" ,"./"]

ENTRYPOINT pipenv shell 

RUN python3 main.py
