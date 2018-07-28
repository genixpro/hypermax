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
            }
        }
    }

Next, define how you want to execute your optimization function:

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


# Detailed Configuration

## Loss / Cost Functions

We support several different types of cost functions.

## Timing Loss

You can include the time your model takes to train as one of your loss functions as well.  This makes it convenient 
to teach the algorithm to avoid bad hyper parameters which lead to long training times. Many algorithms have poor
combinations of parameters which can lead to long execution time with no improvement in performance.

If the algorithm takes less then the target_time, then no penalty is incurred. As the time taken goes between
target_time and max_time, the penalty is introduced quadratically. At max_time, the penalty is exactly
penalty_at_max.

This usually results in the algorithm choosing a value between target_time and max_time, but closer
to target_time. For example, with the following:

    {
        "metrics": {
            "time": {
                "type": "time",
                "target_time": 5,
                "max_time": 10,
                "penalty_at_max": 0.1
            }
        }
    }

If the algorithm takes 5.0 seconds, no penalty is introduced. At 6.0 seconds, the penalty is:
    
    = ((6 - 5) ^ 2 / (10 - 5)^2)*0.1
    = 0.0025

At 9 seconds, the penalty is:

    = ((9 - 5) ^ 2 / (10 - 5) ^ 2)*0.1
    = 0.064

At 10 seconds, the penalty is:

    = ((10 - 5) ^ 2 / (10 - 5) ^ 2)*0.1
    = 0.1

Longer times will have even larger penalties.
