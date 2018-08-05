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
 
Hypermax will automatically construct a loss function which optimizes your metrics, based on the bounds and targets
that you provide. It will then begin the hyper parameter search.


# Detailed Configuration


## Model Execution

There are several different ways of executing your model.

### Python Functions

The most straight forward way to execute your model is by defining a Python function. To do this, simply provide the 
name of the module and the name of the function in the "module" and "name" functions, like so:

    {
        "function": {
            "type": "python_function",
            "module": "model",
            "name": "trainModel"
        }
    }

Remember that you do not include the extension of the name of your module, there is no ".py" on it. The module is
referenced using Pythons standard system. This means that you can directly reference any files in the current working
directory simply by their file-name. Alternatively, you can reference a system-package or a Python package that is
setup elsewhere. As long as this works:

    $ python3
    
    Python 3.6.5 (default, Mar 29 2018, 18:20:46) 
    [GCC 8.0.1 20180317 (Red Hat 8.0.1-0.19)] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import module_name
    >>> module.foobar()
    
Then this will to:

    {
        "function": {
            "type": "python_function",
            "module": "module_name",
            "name": "foobar"
        }
    }

### Format of the result

The results can be provided in one of two formats. The simplest is to just return the loss directly as a single floating point value
from your cost function, or print it to standard output in your executable. For example:

    def trainModel(parameters):
        # Do some fany stuff
        loss = 1.0
        return loss

or as an executable:

    #!/usr/bin/python3

    # Do some fany stuff
    loss = 1.0
    print(loss)

If you are using multiple losses though, you will have to return each of them as part of a JSON object. For example:

    def trainModel(parameters):
        # Do some fany stuff
        accuracy = 0.9
        stddev = 0.1
        return {"accuracy": accuracy, "stddev": stddev}

or as an executable:

    #!/usr/bin/python3

    import json

    # Do some fany stuff
    accuracy = 0.9
    stddev = 0.1
    print(json.dumps({"accuracy": accuracy, "stddev": stddev}))

If you want to store additional metadata with your model, you can. Any fields that are unrecognized for any other purpose will be automatically considered as metadata.

    def trainModel(parameters):
        # Do some fany stuff
        loss = 1.0
        additional_statistic = 42.0
        return {"loss": loss, "additional_statistic": additional_statistic}

The time your model takes is automatically measured by Hypermax (time can be used for punishing your model for taking too long, see Losses section).
However, you may only care about the execution / run-time of your model, and not about the training time. In these cases, you can return "time" as
an additional variable.

    def trainModel(parameters):
        # Do some fany stuff
        model = Model()
        model.train()
        start = datetime.now()
        loss = model.test()
        end = datetime.now()
        return {"loss": loss, "time": (end-start).total_seconds()}

It should be noted that this time is not the same time used for auto_kill purposes. This is the time that will be showed in the UI and considered for optimization
purposes.

### Automatically killing models due to running time or RAM usage

Sometimes, your models may be behaving very poorly in certain parts of your hyper-parameter space. It is thus possible,
and indeed recommended, to set add limits on how long your model can be running for and how much RAM it can use. This
prevents your optimization routine from getting hung due to a model that takes too long to train, or crashing entirely
because it uses too much RAM.

To do this, simply add in a auto_kill_max_time, auto_kill_max_ram, or auto_kill_max_system_ram option, and set a 
a kill_loss variable to indicate what the loss should be for models which are killed.

auto_kill_max_time is specified in seconds. auto_kill_max_ram and auto_kill_max_system_ram are both specified in
megabytes, the kind which are based by 1024 (not 1000).

auto_kill_max_ram only measures the RAM of the model process. However, if your cost-function has other various
sub-processes which take up RAM, these will not be counted. Therefore, you can use auto_kill_max_system_ram in
these cases to prevent total system RAM usage from creeping too high (the assumption being your model is what is
taking up the systems RAM). You are able to provide both at the same time (if you want to).

auto_kill_loss is just a floating point indicating the total loss that should be given to the optimizer when the model
is killed. This helps teach the optimizer to avoid hyper-parameters which lead to models being killed.

    {
        "function": {
            "type": "python_function",
            "module": "model",
            "name": "trainModel",
            "auto_kill_max_time": 120.0,
            "auto_kill_max_ram": 512,
            "auto_kill_max_system_ram": 3800,
            "auto_kill_loss": 1.0
        }
    }

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
    - Able to configure save-location.
    - Automatic uploading of results to Google Drive
- Model Execution
    - Autokill models that take too much GPU RAM
    - Autokill models that take too much Disk / Network (not sure about this one)
    - Fix bug related to using too many file handlers.
    - Automatically keep logs / errors for models executing
    - Execute model as an executable
    - Execute model remotely, through ssh
    - Execute model by sending a message through message-buffer like RabbitMQ or Kafka (receive results same way)
    - Rsync a folder prior to remote execution
    - Can attach additional arbitrary metadata to your model results
    - Able to have "speed-tests" on models, where your hyper-parameters are tested on a reduced dataset in order to measure the speed. Useful to eliminate bad hyper-parameters without executing the full model.
    - Similarly, able to run "memory tests" on models, ensuring your hyper-parameters don't lead to excessive ram usage
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
    - Able to monitor the RAM (and GPU RAM) usage of currently executing models
    - Able to monitor the disk usage of currently executing models
    - Able to monitor the network usage of currently executing models
    - Able to monitor general system stats, such as CPU, network, disk, and ram
- Losses:
    - Able to have multiple weighted loss functions
    - Automatically pass loss-function through math-function based on min,max,target to add in asymptotes at target values
    - Write documentation related to how your cost-function is manipulated to improve results
    - Convenient way to add Time as an additional loss on your model
    - Time computed automatically, but can be overridden if provided in results
    - Can view all the losses for a given model in the UI, not just final loss
    - Convenient way to add in peak or median RAM (and GPU RAM) as an additional loss on your model
    - Convenient way to add in disk / network usage as a additional loss on your model
- Tools for Hyperparameter Tuning:
    - View a hotspot grid between two hyper-parameters, to see the sensitive areas
    - Improve feature-vector design for hyper-parameter correlations
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

