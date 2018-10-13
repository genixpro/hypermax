import hyperopt
import datetime
from hypermax.hyperparameter import Hyperparameter
import json

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

        parameters = Hyperparameter(hyperparameterSpace).getFlatParameters()

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

            for key in result.keys():
                if key not in self.resultInformationKeys:
                    matchingParameters = [parameter for parameter in parameters if parameter.name == key]
                    if len(matchingParameters)==0:
                        raise ValueError("Our hyperparameter search space did not contain a " + key + " parameter.")

                    parameter = matchingParameters[0]

                    value = result[key]
                    if value is not "":
                        if 'enum' in parameter.config:
                            data['misc']['idxs']['root.' + key] = [resultIndex]
                            data['misc']['vals']['root.' + key] = [parameter.config['enum'].index(value)]
                        elif parameter.config['type'] == 'number':
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