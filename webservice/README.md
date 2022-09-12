
# build a docker environment 
sudo docker build -t wine-quality-prediction-service:v1 .

# run docker 
sudo docker run -it --rm -p 9696:9696 wine-quality-prediction-service:v1
