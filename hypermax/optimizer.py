import hyperopt
import csv
import json
from pprint import pprint
import datetime
import time
import numpy.random
import random
from hypermax.configuration import Configuration


class Optimizer:
    def __init__(self, configuration):
        self.config = Configuration(configuration)

        self.space = self.config.createHyperparameterSpace()
        self.executor = self.config.createExecutor()

        self.trials = hyperopt.Trials()

    def runOptimization(self):
        best = None
        for n in range(1):
            rstate = numpy.random.RandomState(seed=int(random.randint(1, 2**32-1)))
            best = hyperopt.fmin(fn=lambda params: self.executor.run(params),
                                 space=self.space,
                                 algo=hyperopt.tpe.suggest,
                                 max_evals=100,
                                 trials=self.trials,
                                 rstate=rstate)
            self.exportCSV('results.csv')
            self.importCSV('results.csv')

    resultInformationKeys = [
        'trial',
        'status',
        'loss'
    ]

    def convertTrialsToResults(self):
        results = []
        for trial in self.trials.trials:
            data = {
                "trial": trial['tid'],
                "status": trial['result']['status'],
                "loss": trial['result']['loss']
            }

            params = trial['misc']['vals']
            for key in params.keys():
                if len(params[key]) == 1:
                    data[key[5:]] = json.dumps(params[key][0])
                else:
                    data[key[5:]] = ''

            results.append(data)
        return results

    def convertResultsToTrials(self, results):
        self.trials = hyperopt.Trials()

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
                    value = result[key]
                    if value is not "":
                        data['misc']['idxs']['root.' + key] = [resultIndex]
                        data['misc']['vals']['root.' + key] = [json.loads(value)]
                    else:
                        data['misc']['idxs']['root.' + key] = []
                        data['misc']['vals']['root.' + key] = []

            self.trials.insert_trial_doc(data)

    def exportCSV(self, fileName):
        results = self.convertTrialsToResults()

        with open(fileName, 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys(), dialect='unix')
            writer.writeheader()
            writer.writerows(results)

    def importCSV(self, fileName):
        with open(fileName, 'rt') as file:
            reader = csv.DictReader(file, dialect='unix')
            rows = list(reader)
        self.convertResultsToTrials(rows)
