import hyperopt
import csv
import json
from pprint import pprint
import datetime
import time
import numpy.random
import threading
import queue
import random
import concurrent.futures
import functools
from hypermax.hyperparameter import Hyperparameter
import sklearn.covariance

from hypermax.configuration import Configuration


class Optimizer:
    resultInformationKeys = [
        'trial',
        'status',
        'loss',
        'time',
        'log'
    ]

    def __init__(self, configuration):
        self.config = Configuration(configuration)

        self.space = self.config.createHyperparameterSpace()
        self.executor = self.config.createExecutor()

        self.threadExecutor = concurrent.futures.ThreadPoolExecutor()

        self.results = []

        self.best = None
        self.bestLoss = None

        self.thread = threading.Thread(target=lambda: self.optimizationThread(), daemon=True)

        self.totalTrials = 100

    def __del__(self):
        self.threadExecutor.shutdown(wait=True)

    def completed(self):
        return len(self.results)

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

        def testSample(params, trial):
            start = datetime.datetime.now()
            modelResult = self.executor.run(parameters=params)
            end = datetime.datetime.now()

            result = {}
            result['trial'] = trial

            if 'loss' in modelResult:
                result['loss'] = modelResult['loss']
            elif 'accuracy' in modelResult:
                result['loss'] = modelResult['accuracy']

            if 'status' in modelResult:
                result['status'] = modelResult['status']
            else:
                result['status'] = 'ok'

            if 'log' in modelResult:
                result['log'] = modelResult['log']
            else:
                result['log'] = ''

            result['time'] = (end-start).total_seconds()

            def recurse(key, value, root):
                result_key = root + "." + key
                if isinstance(value, str):
                    result[result_key[1:]] = value
                elif isinstance(value, float) or isinstance(value, bool) or isinstance(value, int):
                    result[result_key[1:]] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        recurse(subkey, subvalue, result_key)

            for key in params.keys():
                value = params[key]
                recurse(key, value, '')
            return result

        futures = []
        for index, sample in enumerate(samples):
            futures.append(self.threadExecutor.submit(testSample, sample, len(self.results) + index))

        concurrent.futures.wait(futures)

        results = [future.result() for future in futures]

        for result in results:
            if self.best is None or result['loss'] < self.bestLoss:
                self.bestLoss = result['loss']
                self.best = result

        self.results = self.results + results

    def runOptimization(self):
        self.thread.start()

    def optimizationThread(self):
        while len(self.results) < self.totalTrials:
            self.runOptimizationRound()

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
                        data['misc']['vals']['root.' + key] = [value]
                    else:
                        data['misc']['idxs']['root.' + key] = []
                        data['misc']['vals']['root.' + key] = []

            trials.insert_trial_doc(data)
        return trials

    def exportCSV(self, fileName):
        with open(fileName, 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys(), dialect='unix')
            writer.writeheader()
            writer.writerows(self.results)

    def importCSV(self, fileName):
        with open(fileName, 'rt') as file:
            reader = csv.DictReader(file, dialect='unix')
            rows = list(reader)
        self.convertResultsToTrials(rows)

    def computeCorrelations(self):
        inputs = []

        keys = Hyperparameter(self.config.data['hyperparameters']).getFlatParameterNames()

        values = {}
        types = {}
        for key in keys:
            values[key] = set()
            types[key] = set()

        for result in self.results:
            for key in keys:
                value = result[key[5:]]
                values[key].add(value)
                types[key].add(type(value).__name__)

        vectors = []
        labels = []
        outputs = []
        for result in self.results:
            vector = []
            vectorLabels = []
            for key in keys:
                value = result[key[5:]]
                if 'bool' in types[key] or 'int' in types[key] or 'float' in types[key]:
                    if isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                        vector.append(float(value))
                        vectorLabels.append(key + ".number")
                    else:
                        vector.append(-1)
                        vectorLabels.append(key + ".number")
                if 'NoneType' in types[key]:
                    if value is None:
                        vector.append(value)
                        vectorLabels.append(key + ".none")
                    else:
                        vector.append(-1)
                        vectorLabels.append(key + ".none")
                if 'str' in types[key]:
                    classes = [v for v in values[key] if isinstance(v, str)]

                    if isinstance(value, str):
                        for v in classes:
                            if value == v:
                                vector.append(1.0)
                            else:
                                vector.append(0.0)
                            vectorLabels.append(key + ".class." + v)
                    else:
                        for v in classes:
                            vector.append(0)
                            vectorLabels.append(key + ".class." + v)
            vectors.append(vector)
            outputs.append(result['loss'])
            if not labels:
                labels = vectorLabels

        model = sklearn.covariance.LedoitWolf()
        model.fit(numpy.array(vectors), numpy.array(outputs))

        covariances = model.covariance_
        correlations = numpy.zeros_like(covariances)

        deviations = numpy.std(vectors, axis=0)

        for label1Index in range(len(labels)):
            for label2Index in range(len(labels)):
                correlations[label1Index][label2Index] = covariances[label1Index][label2Index] / (deviations[label1Index] * deviations[label2Index])

        return correlations, labels



