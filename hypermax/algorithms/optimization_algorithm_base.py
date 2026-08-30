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

        selectors = []

        def collectSelectors(node, path=""):
            alternatives = node.get("anyOf", node.get("oneOf"))
            if alternatives is not None:
                if path:
                    selectors.append((path, node.get("name", "root." + path)))
                for index, child in enumerate(alternatives):
                    childPath = f"{path}.{index}" if path else str(index)
                    collectSelectors(child, childPath)
            elif node.get("type") == "object":
                for key, child in node.get("properties", {}).items():
                    childPath = f"{path}.{key}" if path else key
                    collectSelectors(child, childPath)

        collectSelectors(hyperparameterSpace)

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
                value = result.get(param.name)
                if value != "" and value is not None:
                    if 'enum' in param.config:
                        value = param.config['enum'].index(value)

                    data['misc']['idxs'][param.hyperoptVariableName] = [resultIndex]
                    data['misc']['vals'][param.hyperoptVariableName] = [value]
                else:
                    data['misc']['idxs'][param.hyperoptVariableName] = []
                    data['misc']['vals'][param.hyperoptVariableName] = []

            # ``getFlatParameters`` historically excludes hp.choice selector
            # nodes.  Hyperopt still requires their observations when trials
            # from conditional spaces are reconstructed.
            for resultName, hyperoptName in selectors:
                value = result.get(resultName)
                if value != "" and value is not None:
                    data['misc']['idxs'][hyperoptName] = [resultIndex]
                    data['misc']['vals'][hyperoptName] = [int(value)]
                else:
                    data['misc']['idxs'][hyperoptName] = []
                    data['misc']['vals'][hyperoptName] = []

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
