import csv
import json
import os.path
from pprint import pprint
import datetime
import time
import threading
import copy
import tempfile
import subprocess
import concurrent.futures
import functools
import atexit
from hypermax.execution import Execution
from hypermax.hyperparameter import Hyperparameter
from hypermax.results_analyzer import ResultsAnalyzer
from hypermax.algorithms.human_guided_optimizer_wrapper import HumanGuidedOptimizerWrapper
from hypermax.algorithms.tpe_optimizer import TPEOptimizer
from hypermax.algorithms.random_search_optimizer import RandomSearchOptimizer

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

        method = self.searchConfig.get("method")
        if method not in {"random", "tpe", "atpe", "atpe2", "abh", "abh2"}:
            raise ValueError("search.method must be one of: random, tpe, atpe, atpe2, abh, abh2")

        iterations = self.searchConfig.get("iterations")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise ValueError("search.iterations must be a positive integer")

        seed = self.searchConfig.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("search.seed must be an integer")
        self.seed = seed

        parallel = configuration.get("function", {}).get("parallel", 1)
        if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel < 1:
            raise ValueError("function.parallel must be a positive integer")

        self.space = self.config.createHyperparameterSpace()

        self.threadExecutor = concurrent.futures.ThreadPoolExecutor()

        self.resultsAnalyzer = ResultsAnalyzer(configuration)

        self.results = []
        self.resultFutures = []

        self.best = None
        self.bestLoss = None

        self.thread = threading.Thread(target=lambda: self.optimizationThread(), daemon=True if configuration.get("ui", {}).get("enabled", True) else False)

        self.totalTrials = iterations
        self.trialsSinceResultsUpload = None
        self.resultsExportFuture = None

        self.currentTrials = []
        self.allWorkers = set(range(parallel))
        self.occupiedWorkers = set()
        self.trialNumber = 0

        self.lastATPEParameters = None
        self.lastLockedParameters = None
        self.atpeParamDetails = None

        self.tpeOptimizer = TPEOptimizer(seed=seed)
        self.randomSearchOptimizer = RandomSearchOptimizer(seed=seed)
        self.atpeOptimizer = None
        self.abhOptimizer = None

        if method in {"atpe", "abh"}:
            try:
                from hypermax.algorithms.atpe_optimizer import ATPEOptimizer
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "Adaptive-TPE could not load LightGBM. On macOS, install the "
                    "OpenMP runtime with `brew install libomp`, then retry."
                ) from exc

            self.atpeOptimizer = ATPEOptimizer(seed=seed)
            if method == "abh":
                from hypermax.algorithms.adaptive_bayesian_hyperband_optimizer import AdaptiveBayesianHyperband

                self.abhOptimizer = AdaptiveBayesianHyperband(
                    self.atpeOptimizer,
                    self.searchConfig.get("min_budget", 1),
                    self.searchConfig.get("max_budget", 100),
                    self.searchConfig.get("eta", 3),
                    seed=seed,
                )
        elif method in {"atpe2", "abh2"}:
            from hypermax.algorithms.atpe2_optimizer import ATPE2Optimizer

            self.atpeOptimizer = ATPE2Optimizer(seed=0 if seed is None else seed)
            if method == "abh2":
                from hypermax.algorithms.adaptive_bayesian_hyperband_optimizer import AdaptiveBayesianHyperband

                self.abhOptimizer = AdaptiveBayesianHyperband(
                    self.atpeOptimizer,
                    self.searchConfig.get("min_budget", 1),
                    self.searchConfig.get("max_budget", 100),
                    self.searchConfig.get("eta", 3),
                    seed=seed,
                )

        self.humanGuidedATPEOptimizer = HumanGuidedOptimizerWrapper(
            self.atpeOptimizer or self.tpeOptimizer
        )

    def __del__(self):
        threadExecutor = getattr(self, "threadExecutor", None)
        if threadExecutor:
            threadExecutor.shutdown(wait=True)


    @classmethod
    def configurationSchema(self):
        """ This method returns the configuration schema for the optimization module. The schema
            is a standard JSON-schema object."""
        return {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ['atpe', 'atpe2', 'tpe', 'random', 'abh', 'abh2']},
                "iterations": {"type": "integer", "minimum": 1},
                "budget": {"type": "number"},
                "seed": {"type": "integer"}
            },
            "required": ['method', 'iterations']
        }

    def completed(self):
        return len(self.results)

    def sampleNext(self):
        if self.searchConfig['method'] == 'tpe':
            return self.tpeOptimizer.recommendNextParameters(self.config.data['hyperparameters'], self.results, self.currentTrials)
        elif self.searchConfig['method'] == 'random':
            return self.randomSearchOptimizer.recommendNextParameters(self.config.data['hyperparameters'], self.results, self.currentTrials)
        elif self.searchConfig['method'] == 'atpe':
            params = self.humanGuidedATPEOptimizer.recommendNextParameters(self.config.data['hyperparameters'], self.results, self.currentTrials)
            self.lastATPEParameters = self.atpeOptimizer.lastATPEParameters
            self.lastLockedParameters = self.atpeOptimizer.lastLockedParameters
            self.atpeParamDetails = self.atpeOptimizer.atpeParamDetails
            return params
        elif self.searchConfig['method'] == 'atpe2':
            params = self.atpeOptimizer.recommendNextParameters(self.config.data['hyperparameters'], self.results, self.currentTrials)
            self.lastATPEParameters = self.atpeOptimizer.lastATPEParameters
            self.lastLockedParameters = self.atpeOptimizer.lastLockedParameters
            self.atpeParamDetails = self.atpeOptimizer.atpeParamDetails
            return params
        elif self.searchConfig['method'] == 'abh':
            params = self.abhOptimizer.recommendNextParameters(self.config.data['hyperparameters'], self.results, self.currentTrials)
            self.lastATPEParameters = self.atpeOptimizer.lastATPEParameters
            self.lastLockedParameters = self.atpeOptimizer.lastLockedParameters
            self.atpeParamDetails = self.atpeOptimizer.atpeParamDetails
            return params
        elif self.searchConfig['method'] == 'abh2':
            params = self.abhOptimizer.recommendNextParameters(self.config.data['hyperparameters'], self.results, self.currentTrials)
            self.lastATPEParameters = self.atpeOptimizer.lastATPEParameters
            self.lastLockedParameters = self.atpeOptimizer.lastLockedParameters
            self.atpeParamDetails = self.atpeOptimizer.atpeParamDetails
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

        def testSample(params, trial, worker, currentTrial):
            start = datetime.datetime.now()
            execution = Execution(self.config.data['function'], parameters=params, worker_n=worker)
            modelResult = execution.run()
            end = datetime.datetime.now()

            result = Hyperparameter(self.config.data['hyperparameters']).convertToFlatValues(params)

            for key in params.keys():
                if key.startswith("$"):
                    result[key] = params[key]

            result['trial'] = trial
            self.resultsAnalyzer.makeDirs(os.path.join(self.resultsAnalyzer.directory, "logs"))

            if 'loss' in modelResult:
                result['loss'] = modelResult['loss']
            elif 'accuracy' in modelResult:
                result['loss'] = modelResult['accuracy']

            if 'status' in modelResult:
                result['status'] = modelResult['status']
            else:
                result['status'] = 'ok'

            if 'log' in modelResult:
                fileName = os.path.join(self.resultsAnalyzer.directory, "logs", "trial_" + str(trial) + ".txt")
                with open(fileName, "wt") as file:
                    file.write(modelResult['log'])
                result['log'] = fileName
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

            self.currentTrials.remove(currentTrial)

            return result

        completionEvent = threading.Event()

        def onCompletion(worker, future):
            try:
                self.occupiedWorkers.remove(worker)

                self.results.append(future.result())

                self.computeCurrentBest()

                if not self.config.data.get("ui", {}).get("enabled", True):
                    pprint(future.result())

                if self.resultsExportFuture is None or (self.resultsExportFuture.done() and len(self.results) > 5):
                    self.resultsExportFuture = self.threadExecutor.submit(
                        lambda: self.outputResultsWithBackup(self.config.data.get("results", {}).get("graphs", True)))
                else:
                    self.outputResultsWithBackup(False)

                if 'hypermax_results' in self.config.data:
                    if self.trialsSinceResultsUpload is None or self.trialsSinceResultsUpload >= self.config.data['hypermax_results']['upload_frequency']:
                        self.saveResultsToHypermaxResultsRepository()
                        self.trialsSinceResultsUpload = 1
                    else:
                        self.trialsSinceResultsUpload += 1
            finally:
                completionEvent.set()

        self.occupiedWorkers.add(sampleWorker)
        currentTrial = {
            "start": datetime.datetime.now(),
            "trial": self.trialNumber,
            "worker": sampleWorker,
            "params": copy.deepcopy(sample),
        }
        self.currentTrials.append(currentTrial)
        sampleFuture = self.threadExecutor.submit(
            testSample, sample, self.trialNumber, sampleWorker, currentTrial
        )
        sampleFuture.add_done_callback(functools.partial(onCompletion, sampleWorker))
        sampleFuture.completionEvent = completionEvent
        self.trialNumber += 1
        return sampleFuture

    def runOptimizationThread(self):
        self.thread.start()

    def outputResultsWithBackup(self, graphs, workers=1):
        self.resultsAnalyzer.outputResultsFolder(self, graphs, workers=workers)
        directory_head, directory_tail = os.path.split(self.resultsAnalyzer.directory)
        backup_directory = os.path.join(directory_head, ".backup_" + directory_tail + "~")
        self.resultsAnalyzer.outputResultsFolder(self, graphs, directory=backup_directory, workers=workers)

    def optimizationThread(self):
        # Make sure we output basic results if the process is killed for some reason.
        backupHandler = lambda: self.outputResultsWithBackup(False)
        atexit.register(backupHandler)

        futures = []
        for worker in range(min(len(self.allWorkers), self.totalTrials - len(self.results))):
            futures.append(self.startOptmizationJob())

        # ``currentTrials`` is populated inside worker threads, so using it to
        # decide whether to submit more work races with fast model functions and
        # can overshoot the configured iteration count. ``trialNumber`` is
        # incremented synchronously when each job is submitted.
        while self.trialNumber < self.totalTrials:
            completedFuture = list(concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)[0])[0]
            completedFuture.completionEvent.wait()
            completedFuture.result()
            futures.remove(completedFuture)
            time.sleep(0.05)
            futures.append(self.startOptmizationJob())

        concurrent.futures.wait(futures)
        for future in futures:
            future.completionEvent.wait()
            future.result()

        if self.resultsExportFuture is not None:
            self.resultsExportFuture.result()

        # We are completed, so we can allocate a full contingent of workers
        self.outputResultsWithBackup(
            self.config.data.get("results", {}).get("graphs", True), workers=4
        )
        atexit.unregister(backupHandler)

    def exportGuidanceJSON(self, fileName):
        with open(fileName, 'wt') as file:
            json.dump(self.humanGuidedATPEOptimizer.guidanceOptions, file, indent=4, sort_keys=True)

    def importGuidanceJSON(self, fileName):
        with open(fileName, 'rt') as file:
            self.humanGuidedATPEOptimizer.guidanceOptions = json.load(file)

    def exportResultsCSV(self, fileName):
        allKeys = set()
        for result in self.results:
            for key in result:
                allKeys.add(key)

        fieldNames = self.resultInformationKeys + sorted(allKeys.difference(set(self.resultInformationKeys))) # Make sure we keep the order of the field names consistent when writing the csv
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
                    if value is not None and value != "":
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
