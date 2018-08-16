import hyperopt
import csv
import json
import os.path
from pprint import pprint
import datetime
import time
import numpy.random
import threading
import queue
import copy
import tempfile
import random
import subprocess
import concurrent.futures
import functools
import atexit
import jsonschema
from hypermax.execution import Execution
from hypermax.hyperparameter import Hyperparameter
from hypermax.results_analyzer import ResultsAnalyzer

from hypermax.configuration import Configuration


class Optimizer:
    resultInformationKeys = [
        'trial',
        'status',
        'loss',
        'time',
        'log',
        'error'
    ]

    def __init__(self, configuration):
        self.config = Configuration(configuration)

        self.searchConfig = configuration.get('search', {})
        jsonschema.validate(self.searchConfig, self.configurationSchema())

        self.space = self.config.createHyperparameterSpace()

        self.threadExecutor = concurrent.futures.ThreadPoolExecutor()

        self.resultsAnalyzer = ResultsAnalyzer(configuration)

        self.results = []
        self.resultFutures = []

        self.best = None
        self.bestLoss = None

        self.thread = threading.Thread(target=lambda: self.optimizationThread(), daemon=True if configuration.get("ui", {}).get("enabled", True) else False)

        self.totalTrials = self.searchConfig.get("iterations")
        self.trialsSinceResultsUpload = None
        self.resultsExportFuture = None

        self.currentTrials = []
        self.allWorkers = set(range(self.config.data['function'].get('parallel', 1)))
        self.occupiedWorkers = set()
        self.trialNumber = 0


    def __del__(self):
        if self.threadExecutor:
            self.threadExecutor.shutdown(wait=True)


    @classmethod
    def configurationSchema(self):
        """ This method returns the configuration schema for the optimization module. The schema
            is a standard JSON-schema object."""
        return {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ['tpe', 'random']},
                "iterations": {"type": "number"},
            },
            "required": ['method', 'iterations']
        }

    def completed(self):
        return len(self.results)

    def sampleNext(self):
        rstate = numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1)))
        params = {}

        def sample(parameters):
            nonlocal params
            params = parameters
            return {"loss": 0.5, 'status': 'ok'}

        trials = self.convertResultsToTrials(self.results)
        if self.searchConfig['method'] == 'tpe':
            hyperopt.fmin(fn=sample,
                          space=self.space,
                          algo=functools.partial(hyperopt.tpe.suggest, n_EI_candidates=4, gamma=0.25),
                          max_evals=1,
                          trials=trials,
                          rstate=rstate)
        elif self.searchConfig['method'] == 'random':
            hyperopt.fmin(fn=sample,
                          space=self.space,
                          algo=hyperopt.rand.suggest,
                          max_evals=1,
                          trials=trials,
                          rstate=rstate)

        return params

    def computeCurrentBest(self):
        best = None
        bestLoss = None
        for result in self.results:
            if (best is None and result['loss'] is not None ) or (result['loss'] is not None and result['loss'] < bestLoss):
                best = result
                bestLoss = result['loss']
        self.best = best
        self.bestLoss = bestLoss


    def startOptmizationJob(self):
        availableWorkers = list(sorted(self.allWorkers.difference(self.occupiedWorkers)))

        sampleWorker = availableWorkers[0]
        sample = self.sampleNext()

        def testSample(params, trial, worker):
            currentTrial = {
                "start": datetime.datetime.now(),
                "trial": trial,
                "worker": worker,
                "params": copy.deepcopy(params)
            }
            self.currentTrials.append(currentTrial)
            start = datetime.datetime.now()
            execution = Execution(self.config.data['function'], parameters=params, worker_n=worker)
            modelResult = execution.run()
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

            if 'error' in modelResult:
                result['error'] = modelResult['error']
            else:
                result['error'] = ''

            if 'time' in modelResult:
                result['time'] = modelResult['time']
            else:
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

            self.currentTrials.remove(currentTrial)

            return result

        def onCompletion(worker, future):
            self.occupiedWorkers.remove(worker)

            self.results.append(future.result())

            self.computeCurrentBest()

            if self.resultsExportFuture is None or (self.resultsExportFuture.done() and len(self.results) > 5):
                self.resultsExportFuture = self.threadExecutor.submit(
                    lambda: self.resultsAnalyzer.outputResultsFolder(self, self.config.data.get("results", {}).get("graphs", True)))
            else:
                self.resultsAnalyzer.outputResultsFolder(self, False)

            if 'hypermax_results' in self.config.data:
                if self.trialsSinceResultsUpload is None or self.trialsSinceResultsUpload >= self.config.data['hypermax_results']['upload_frequency']:
                    self.saveResultsToHypermaxResultsRepository()
                    self.trialsSinceResultsUpload = 1
                else:
                    self.trialsSinceResultsUpload += 1

        self.occupiedWorkers.add(sampleWorker)
        sampleFuture = self.threadExecutor.submit(testSample, sample, self.trialNumber, sampleWorker)
        sampleFuture.add_done_callback(functools.partial(onCompletion, sampleWorker))
        self.trialNumber += 1
        return sampleFuture

    def runOptimization(self):
        self.thread.start()

    def optimizationThread(self):
        # Make sure we output basic results if the process is killed for some reason.
        atexit.register(lambda: self.resultsAnalyzer.outputResultsFolder(self, False))

        futures = []
        for worker in range(min(len(self.allWorkers), self.totalTrials - len(self.results))):
            futures.append(self.startOptmizationJob())

        while (len(self.results) + len(self.currentTrials)) < self.totalTrials:
            completedFuture = list(concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)[0])[0]
            futures.remove(completedFuture)
            time.sleep(0.05)
            futures.append(self.startOptmizationJob())

        concurrent.futures.wait(futures)

        # We are completed, so we can allocate a full contingent of workers
        self.resultsAnalyzer.outputResultsFolder(self, True, workers=4)

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

        parameters = Hyperparameter(self.config.data['hyperparameters']).getFlatParameters()

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
                    matchingParameters = [parameter for parameter in parameters if parameter.root[5:] == key]
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



    def exportResultsCSV(self, fileName):
        fieldNames = self.resultInformationKeys + sorted(set(self.results[0].keys()).difference(set(self.resultInformationKeys))) # Make sure we keep the order of the field names consistent when writing the csv
        with open(fileName, 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=fieldNames if len(self.results) > 0 else [], dialect='unix')
            writer.writeheader()
            writer.writerows(self.results)

    def importResultsCSV(self, fileName):
        with open(fileName) as file:
            reader = csv.DictReader(file)
            results = list(reader)
            newResults = []
            for result in results:
                newResult = {}
                for key,value in result.items():
                    if value:
                        try:
                            if '.' in value or 'e' in value:
                                newResult[key] = float(value)
                            else:
                                newResult[key] = int(value)
                        except ValueError:
                            newResult[key] = value
                    elif key == 'loss':
                        newResult[key] = None
                    elif key == 'log':
                        newResult[key] = ''
                    else:
                        newResult[key] = None
                newResults.append(newResult)
            self.results = newResults
        self.computeCurrentBest()
        self.trialNumber = len(self.results)

    def saveResultsToHypermaxResultsRepository(self):
        try:
            hypermaxResultsConfig = self.config.data['hypermax_results']
            with tempfile.TemporaryDirectory() as directory:
                process = subprocess.run(['git', 'clone', 'git@github.com:electricbrainio/hypermax-results.git'], cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                hypermaxResultsDirectory = os.path.join(directory, 'hypermax-results', hypermaxResultsConfig['name'])
                self.resultsAnalyzer.outputResultsFolder(self, detailed=False, directory=hypermaxResultsDirectory)
                with open(os.path.join(hypermaxResultsDirectory, "metadata.json"), 'wt') as file:
                    json.dump(self.config.data['hypermax_results'], file, indent=4)
                process = subprocess.run(['git', 'add', hypermaxResultsDirectory], cwd=os.path.join(directory, 'hypermax-results'))
                process = subprocess.run(['git', 'commit', '-m', 'Hypermax automatically storing results for model ' + hypermaxResultsConfig['name'] + ' with ' + str(len(self.results)) + " trials."], cwd=os.path.join(directory, 'hypermax-results'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process = subprocess.run(['git push'], cwd=os.path.join(directory, 'hypermax-results'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        except Exception as e:
            print(e)
