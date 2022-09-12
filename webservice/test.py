import requests

wine = {
    'fixed acidity': 8, 
    'volatile acidity': 0.17, 
    'citric acid' : 0.4, 
    'residual sugar' : 1.5,
    'chlorides' : 0.05, 
    'free sulfur dioxide' : 18, 
    'total sulfur dioxide' : 50, 
    'density' : 0.99,
    'pH' : 3.3, 
    'sulphates' : 0.6, 
    'alcohol' : 12, 
    'type' : 'red'

}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=wine)
print(response.json())
