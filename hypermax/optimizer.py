import hyperopt
import csv
import json
from pprint import pprint
import datetime
import time
import numpy.random
import random
import concurrent.futures
import functools
from hypermax.configuration import Configuration


class Optimizer:
    resultInformationKeys = [
        'trial',
        'status',
        'loss'
    ]

    def __init__(self, configuration):
        self.config = Configuration(configuration)

        self.space = self.config.createHyperparameterSpace()
        self.executor = self.config.createExecutor()

        self.threadExecutor = concurrent.futures.ThreadPoolExecutor()

        self.results = []

        self.best = None
        self.bestLoss = None

    def __del__(self):
        self.threadExecutor.shutdown(wait=True)

    def sampleNext(self):
        rstate = numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1)))
        params = {}

        def sample(parameters):
            nonlocal params
            params = parameters
            return {"loss": 0, 'status': 'ok'}

        hyperopt.fmin(fn=sample,
                      space=self.space,
                      algo=functools.partial(hyperopt.tpe.suggest, n_EI_candidates=24, gamma=0.25),
                      max_evals=1,
                      trials=self.convertResultsToTrials(self.results),
                      rstate=rstate)

        return params

    def runOptimizationRound(self):
        jobs = self.config.data['function'].get('parallel', 4)
        samples = [self.sampleNext() for job in range(jobs)]

        def testSample(params, trial=-1):
            modelResult = self.executor.run(parameters=params)

            result = {}
            # result['trial'] = trial
            result['loss'] = modelResult['accuracy']
            result['status'] = 'ok'

            for key in params.keys():
                result[key] = json.dumps(params[key])
            return result

        futures = []
        for index, sample in enumerate(samples):
            futures.append(self.threadExecutor.submit(testSample, sample, index))

        concurrent.futures.wait(futures)

        results = [future.result() for future in futures]

        for result in results:
            if self.best is None or result['loss'] < self.bestLoss:
                self.bestLoss = result['loss']
                self.best = result

        self.results = self.results + results

    def runOptimization(self):
        count = 0
        while count < 10000:
            self.runOptimizationRound()
            count += 1

            if self.best['loss'] < 0.2:
                break

    def convertTrialsToResults(self, trials):
        results = []
        for trial in trials.trials:
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

            for key in result.keys():
                if key not in self.resultInformationKeys:
                    value = result[key]
                    if value is not "":
                        data['misc']['idxs']['root.' + key] = [resultIndex]
                        data['misc']['vals']['root.' + key] = [json.loads(value)]
                    else:
                        data['misc']['idxs']['root.' + key] = []
                        data['misc']['vals']['root.' + key] = []

            trials.insert_trial_doc(data)
        return trials

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
