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



# Todo & Wishlist

Feel free to contribute! Reach out to Brad at brad@electricbrain.io or here on Github. We welcome additional contributors.

This is the grand-todo list and was created on August 4, 2018. Some items may have been completed.

- Results
    - Automatic saving of results file, able to configure save-location.
    - Results files have a name which is automatically incremented. Old results never overwritten unless explicitly told too.
    - Automatically export the "results bundle" to a directory (more then just results file)
    - Automatic uploading of results to Google Drive
- Model Execution
    - Autokill long-running models
    - Automatically keep logs / errors for models executing
    - Execute model as an executable
    - Execute model as a python function
    - Execute model remotely, through ssh
    - Execute model by sending a message through message-buffer like RabbitMQ or Kafka (receive results same way)
    - Support different type of formats for the result (e.g. a single number through to json object and more)
    - Rsync a folder prior to remote execution
    - Can attach additional arbitrary metadata to your model results
    - Able to have "speed-tests" on models, where your hyper-parameters are tested on a reduced dataset in order to measure the speed. Useful to eliminate bad hyper-parameters without executing the full model.
    - Able to automatically run additional cross-folds on your best-models, to ensure they aren't statistical flukes
    - Hyper-parameters with rounding set to 1 should be automatically converted to integer
- Configuration:
    - JSON-schema for the configuration files
    - validation of json-schemas
    - Ability to accept yaml as well as JSON
    - Able to have fixed values inside of the hyper-parameter schemas.
    - Able to have "unbounded" hyperparameters (only when using iterative optimization, since TPE doesn't inherently do this)
    - Ability to have hyper-parameters within arrays, such as a list of layers
- Reliability:
    - Able to restart tuning where left off if hypermax crashes
    - Hypermax saves results / reloads from where it left off by default
    - Try to lock in package versions related to Hyperopt so people don't have problems on installation
    - Need a better way of handling exceptions that happen in UI code
    - Execution threads should only communicate through queues, eliminate shared variables (and put locks in the ones we can't eliminate)
    - Control random seed and ensure that runs are reproducible
- General User Interface:
    - Change User-Interface code to use proper organized classes and not ad-hoc style like it is currently
    - View currently running models
    - View recently trained models
    - View models which had errors
    - Fix UI issues related to data-tables (such as in hyperparameter correlations)
    - Able to adjust min,max,smoothing, and domain of the loss chart
    - Move status information to top-right.
    - Predict model execution time based on hyper-parameters and prior data
    - Progress-bar on model training
    - Can change the file-name when exporting the hyper-parameter correlations
    - Can instantly open files which are exported, using xdg-open & equivalents
    - Widget which allows easily viewing/scrolling a large block of text.
    - View the raw hyper-parameter search configuration, as JSON
    - Exporting the hyper-parameters should save them in the exact format they are fed into the model, not in a flattened structure
    - View the hyper-parameter space while model is running
    - Can view arbitrary metadata that was attached to models
    - Make the UI responsive to different console sizes (like a grid system)
    - Fix the bug where the UI doesn't automatically resize when terminal resizes
    - Access the UI through a web-browser
    - Password-protection on web-browser UI
- Losses:
    - Able to have multiple weighted loss functions
    - Automatically pass loss-function through math-function based on min,max,target to add in asymptotes at target values
    - Write documentation related to how your cost-function is manipulated to improve results
    - Convenient way to add Time as an additional loss on your model
    - Time computed automatically, but can be overridden if provided in results
    - Can view all the losses for a given model in the UI, not just final loss
- Tools for Hyperparameter Tuning:
    - View a hotspot grid between two hyper-parameters, to see the sensitive areas
    - Improve feature-vector design for hyper-parameter correlations
    - Export loss-chart to image with matplotlib
    - Export hyper-parameter correlations to image with matplotlib
    - Export hyper-parameter accuracies to image with matplotlib
    - Export hyper-parameter hot-spot grid to image with matplotlib
    - Edit / change the hyper-parameter space while the model is running
    - Estimate the cardinality of the search-space
    - Estimate number of runs per parameter (or something like this)
    - Ability to fit a hyper-model and do a simulated extension of your hyper-parameter search, trying to predict if there are better values that you haven't searched
    - Ability to use the hyper-model to hold certain hyper-parameters constant, and determine the optimal values for remaining hyper-parameters
    - Staged tuning - able to have multiple tuning "stages", which tune only certain hyper-parameters at a time or with different configurations
    - Can have a 'default' value for each of the hyperparameters, e.g. your current best model.
    - Incremental tuning - basically only tunes a handful of hyper-parameters at a time. Can be random or specified
    - Ability to change the hyper parameters for TPE
    - Research some automatic way to guess good TPE hyper-parameters
    - Integrate Bayesian hyper-parameter optimization as an alternative to TPE
    - Integrate grid-search as an alternative to TPE
    - Integrate genetic-algo as an alternative to TPE
- Command-line interface
    - Provide existing results file to continue optimizing existing model (after crash)
    - Execute model without CUI
    - Sample next hyper-parameters to test
    - Export all of the existing types of result-analysis (correlations, hotspot-grids, param vs loss, etc..)
    - Launch web browser UI (without CUI)
    - Write a template configuration file to the current directory
- Library interface:
    - Able to activate hypermax by calling a library function
    - Able to provide our cost-function directly as a python function, rather then just as a JSON description
- Testing
    - Write unit tests for hyper-parameter configuration
    - Write unit tests for model execution
    - Write unit tests for loss functions
    - Write unit tests for the optimizer
    - Write unit tests for the results generation
    - Write unit tests related to reliability (starting / stopping models)
    - Write unit tests for command line interface
    - Write unit tests for web-UI module (just ensure it loads)
    - Write UI tests for the CUI
    - Write end-to-end optimization tests for a few different real datasets / algos
- Template hyperparameter spaces:
    - Create a template hyper-parameter search for lightgbm
    - Create a template hyper-parameter search for xgbboost
    - Create template hyper-parameter searches for various scikit-learn estimators
    - Ability to reference template hyper-parameter searches in your own JSON schemas ($ref)
- Release / Launch:
    - Add in hypermax dependencies to the setup.py, and ensure other configuration in setup.py is working
    - Submit hypermax to pip
    - Test hypermax in following environments, and fix any issues with installation configuration
        - Fresh virtual environment
        - Fresh fedora installation
        - Fresh ubuntu installation
        - macOS
    - Write all the documentation for the README file for Hypermax
    - Create Github wiki, duplicate README documentation in Github Wiki
    - Create Hypermax web-page as sub-domain of Electric Brain, duplicate information from README
    - Write a blog post discussing hypermax and hyper-optimization in general
    - Create optimization examples for:
        - Python
        - Java
        - NodeJS
        - Ruby
        - R
    - Create system packages for:
        - Fedora / Redhat / CentOS
        - Debian / Ubuntu
        - macOS
        - Windows

