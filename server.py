"""server.py

Titanic 生存预测服务
"""

import os
import pickle
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    """预测
    """
    try:
        test_json = request.get_json()
        print(test_json)
        test = np.array([test_json["person"]])
    except Exception as e:
        raise e

    clf = 'model_titanic_v1.pk'

    # 加载模型
    print("Loading the model...")
    loaded_model = None
    with open('model_titanic_v1.pkl','rb') as f:
        loaded_model = pickle.load(f)

    print("The model has been loaded...doing predictions now...")
    predictions = loaded_model.predict(test)
    print("Result: ", predictions)

    responses = jsonify(predictions=int(predictions[0]))
    responses.status_code = 200

    return (responses)
