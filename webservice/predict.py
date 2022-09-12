import joblib
import pandas as pd

from flask import Flask, request, jsonify


pipe = joblib.load('pipeline.pkl')


def predict_json(features):
    X = [features]
    X = pd.DataFrame(X)
    preds = pipe.predict(X)
    return float(preds)


app = Flask('wine-quality-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint(): 
    wine = request.get_json()

    pred = predict_json(wine)

    result = {
        'wine quality': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)