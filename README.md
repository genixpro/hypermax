# Introduction

Hypermax is hyperparameter optimization on steroids. Hypermax wraps the TPE algorithm within Hyperopt
with powerful additional features.

# Installation

Install using pip:

    pip3 install hypermax -g
    
Python3 is required.    

# Getting Started

In Hypermax, you define your hyper-parameter search, including the variables, method of searching, and 
loss functions, using a modified version of JSON schema. In fact, everything is configured through
a JSON object.

Here is an example. Lets say you have the following file, model.py:

    import sklearn.datasets
    import sklearn.ensemble
    import sklearn.metrics
    import datetime
    
    def trainModel(params):
        inputs, outputs = sklearn.datasets.make_hastie_10_2()
        
        startTime = datetime.now()
        
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=int(params['n_estimators']))
        model.fit(inputs, outputs)
        predicted = model.predict(inputs)
        
        finishTime = datetime.now()
        
        auc = sklearn.metrics.auc(outputs, predicted)
        
        return {"auc": auc, "time": (finishTime - startTime).total_seconds()}
        
You configure your hyper parameter search space by defining a JSON-schema object with the needed values:

    {
        "hyperparameters": {
            "type": "object",
            "properties": {
                "n_estimators": {
                    "type": "number",
                    "min": 1,
                    "max": 1000,
                    "scaling": "exponential"
                }
            }
        }
    }

Then you define the metrics you want to optimize

    {
        "metrics": {
            "auc": {
                "min": 0.5,
                "max": 1.0,
                "direction": "increase",
                "scaling": "linear"
            },
            "time": {
                "min": 0.0,
                "direction": "decrease",
                "scaling": "linear"
            }
        }
    }

Define how you want to execute your optimization function:

    {
        "function": {
            "type": "python_function",
            "src": "model.py",
            "name": "trainModel"
        }
    }
    
And lastly, you need to define your hyper parameter search:

    {
        "search": {
            "name": "experiment1",
            "method": "tpe"
        }
    }
    

Pulling it all together, you create a file like this search.json, defining your hyper-parameter search:

    {
        "hyperparameters": {
            "type": "object",
            "properties": {
                "n_estimators": {
                    "type": "number",
                    "min": 1,
                    "max": 1000,
                    "scaling": "exponential"
                }
            }
        },
        "metrics": {
            "auc": {
                "min": 0.5,
                "max": 1.0,
                "direction": "increase",
                "scaling": "linear"
            },
            "time": {
                "min": 0.0,
                "direction": "decrease",
                "scaling": "linear"
            }
        },
        "function": {
            "type": "python_function",
            "src": "model.py",
            "name": "trainModel"
        },
        "search": {
            "name": "experiment1",
            "method": "tpe"
        }
    }
    
And now you can run your hyper-parameter search

    $ hypermax search.json
        -- Hypermax Version 1.0 --
        -- Launching --
    ...  
 
Hypermax will automatically construct a loss function which optimizes your metrics, based on the bounds and targets that you provide. It will then
begin the hyper parameter search.