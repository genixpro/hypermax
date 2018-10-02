import sklearn.datasets
import sklearn.metrics
import math
from datetime import datetime


def trainModel(params):
    inputs, outputs = sklearn.datasets.load_breast_cancer(True)

    startTime = datetime.now()

    targets = {
        "first_value": 383,
        "second_value": 862,
        "third_value": 4,
        "fourth_value": 670,
        "fifth_value": 106,
        "sixth_value": 42,
        "seventh_value": 215,
        "eighth_value": 921,
    }

    accuracy = 0
    total = 0
    for key in targets.keys():
        accuracy += math.sqrt((params[key] - targets[key]) * (params[key] - targets[key]))
        total += targets[key]

    finishTime = datetime.now()

    result = {"accuracy": accuracy/total, "time": (finishTime - startTime).total_seconds()}
    print(result)
    return result


