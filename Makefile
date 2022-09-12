quality_checks:
	isort .
	black .

setup:
	sudo docker build -t wine-quality-prediction-environment:v1 .
	sudo docker run -it --rm -p 9696:9696 wine-quality-prediction-environment:v1
