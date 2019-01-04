import hyperopt
import datetime
from hypermax.hyperparameter import Hyperparameter
import json
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

    def recommendNextParameters(self, hyperparameterSpace, results, lockedValues=None):
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

            trialValues = Hyperparameter(hyperparameterSpace).convertToTrialValues(result)

            for key in trialValues:
                value = trialValues[key]
                if value is not "":
                    data['misc']['idxs']['root.' + key] = [resultIndex]
                    data['misc']['vals']['root.' + key] = [value]
                else:
                    data['misc']['idxs']['root.' + key] = []
                    data['misc']['vals']['root.' + key] = []

            trials.insert_trial_doc(data)
        return trials

    def convertTrialsToResults(self, trials):
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
            for key in params.keys():
                if len(params[key]) == 1:
                    data[key[5:]] = json.dumps(params[key][0])
                else:
                    data[key[5:]] = ''

            results.append(data)
        return results