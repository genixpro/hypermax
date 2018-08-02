import sklearn.datasets
import sklearn.metrics
import math
from datetime import datetime


def trainModel(params):
    inputs, outputs = sklearn.datasets.load_breast_cancer(True)

    startTime = datetime.now()


    # model = sklearn.ensemble.RandomForestClassifier(n_estimators=int(params['n_estimators']))
    # model.fit(inputs, outputs)
    # predicted = model.predict(inputs)

    targets = {
        "first_value": 383,
        "second_value": 862,
        "third_value": 4
    }

    accuracy = 0
    total = 0
    for key in targets.keys():
        accuracy += math.sqrt((params[key] - targets[key]) * (params[key] - targets[key]))
        total += targets[key]

    finishTime = datetime.now()

    # accuracy = sklearn.metrics.accuracy_score(outputs, predicted)

    return {"accuracy": accuracy/total, "time": (finishTime - startTime).total_seconds()}


