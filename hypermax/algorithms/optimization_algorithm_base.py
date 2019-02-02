import hyperopt
import datetime
from hypermax.hyperparameter import Hyperparameter
import json
import copy
from pprint import pprint

class OptimizationAlgorithmBase:
    """ This is the base class for all optimization algorithms. These are the core algorithms which produce
        recommendations on what to try next."""

    resultInformationKeys = [
        'trial',
        'status',
        'loss',
        'time',
        'log',
        'error'
    ]

    def recommendNextParameters(self, hyperparameterSpace, results, currentTrials, lockedValues=None):
        pass



    def convertResultsToTrials(self, hyperparameterSpace, results):
        trials = hyperopt.Trials()

        for resultIndex, result in enumerate(results):
            data = {
                'book_time': datetime.datetime.now(),
                'exp_key': None,
                'misc': {'cmd': ('domain_attachment', 'FMinIter_Domain'),
                         'idxs': {},
                         'tid': resultIndex,
                         'vals': {},
                         'workdir': None},
                'owner': None,
                'refresh_time': datetime.datetime.now(),
                'result': {'loss': result['loss'], 'status': result['status']},
                'spec': None,
                'state': 2,
                'tid': resultIndex,
                'version': 0
            }

            for param in Hyperparameter(hyperparameterSpace).getFlatParameters():
                value = result[param.name]
                if value is not "" and value is not None:
                    if 'enum' in param.config:
                        value = param.config['enum'].index(value)

                    data['misc']['idxs'][param.hyperoptVariableName] = [resultIndex]
                    data['misc']['vals'][param.hyperoptVariableName] = [value]
                else:
                    data['misc']['idxs'][param.hyperoptVariableName] = []
                    data['misc']['vals'][param.hyperoptVariableName] = []

            trials.insert_trial_doc(data)
        return trials

    def convertTrialsToResults(self, hyperparameterSpace, trials):
        results = []
        for trialIndex, trial in enumerate(trials.trials):
            data = {
                "trial": trialIndex,
                "status": trial['result']['status'],
                "loss": trial['result']['loss'],
                "log": "",
                "time": abs((trial['book_time'] - trial['refresh_time']).total_seconds())
            }

            params = trial['misc']['vals']
            for param in Hyperparameter(hyperparameterSpace).getFlatParameters():
                key = param.hyperoptVariableName

                if len(params[key]) == 1:
                    value = params[key][0]
                    if 'enum' in param.config:
                        value = param.config['enum'][value]

                    data[param.name] = value
                else:
                    data[param.name] = ''


            results.append(data)
        return results
