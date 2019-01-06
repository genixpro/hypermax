import numpy as np
import random
from math import log, ceil
from time import time, ctime
from .optimization_algorithm_base import OptimizationAlgorithmBase
from .random_search_optimizer import RandomSearchOptimizer
from pprint import pprint
from ..hyperparameter import Hyperparameter
import hyperopt
import json
import functools
import copy

class AdaptiveBayesianHyperband(OptimizationAlgorithmBase):
    """ This algorithm combines our ATPE optimizer with Hyperband"""

    def __init__(self, baseOptimizer, min_budget, max_budget, eta = 3):
        self.baseOptimizer = baseOptimizer
        self.randomOptimizer = RandomSearchOptimizer()

        self.min_budget = min_budget  # minimum iterations per configuration
        self.max_budget = max_budget  # maximum iterations per configuration

        self.eta = eta  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_budget))
        self.B = (self.s_max + 1) * self.max_budget

        self.results = []  # list of dicts

    @classmethod
    def configurationSchema(self):
        return {}


    def createBudgetSchedule(self):
        skip_last = 0

        runs = []

        for s in reversed( range( self.s_max + 1 )):

            # initial number of configurations
            n = int(ceil(self.B / self.max_budget / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_budget * self.eta ** (-s)

            runs_in_sequence = 0

            for i in range(( s + 1 ) - int( skip_last )):	# changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** ( -i )
                n_budget = r * self.eta ** ( i )

                if n_budget >= self.min_budget:
                    runs.append({
                        "group": s,
                        "round": runs_in_sequence,
                        "configs_start": int(ceil(n_configs)),
                        "configs_finish": int(ceil(n_configs / self.eta)),
                        "input_configs": int(ceil(n_configs * self.eta)),
                        "input_round": runs_in_sequence - 1,
                        "input_budget": -1 if i == 0 else int(ceil(r * self.eta ** ( i - 1 ))),
                        "budget": int(ceil(n_budget))
                    })

                    runs_in_sequence += 1

        return runs
        # return self.results

    def recommendNextParameters(self, hyperparameterSpace, results, lockedValues=None):
        runs = self.createBudgetSchedule()

        results = [result for result in results if result['loss'] is not None]

        # Find which is the largest $loop we find in the results
        if len(results) == 0:
            loop = 0
        else:
            loop = max([result['$loop'] for result in results])

        loopResults = [result for result in results if result['$loop'] == loop]

        # Define which secondary halving runs have enough data to operate
        runsNeeded = []
        for run in runs:
            if run['input_round'] != -1:
                inputResultsForRun = [result for result in loopResults if (result['$group'] == run['group'] and result['$round'] == run['input_round'])]

                if len(inputResultsForRun) < run['input_configs']:
                    continue

            resultsForRun = [result for result in loopResults if (result['$group'] == run['group'] and result['$round'] == run['round'])]

            if len(resultsForRun) < run['configs_start']:
                runsNeeded.append(run)

        runsNeeded = sorted(runsNeeded, key=lambda run: (-run['group'], -run['budget']))

        if len(runsNeeded) == 0:
            runsNeeded = sorted(runs, key=lambda run: run['budget'])
            loop += 1

        run = runsNeeded[0]

        if run['input_round'] == -1:
            resultsForReccomendation = [result for result in results if result['$budget'] == run['budget']]

            if len(resultsForReccomendation) % 5 == 0:
                params = self.randomOptimizer.recommendNextParameters(hyperparameterSpace, resultsForReccomendation)
            else:
                params = self.baseOptimizer.recommendNextParameters(hyperparameterSpace, resultsForReccomendation)

            params['$budget'] = run['budget']
            params['$loop'] = loop
            params['$group'] = run['group']
            params['$round'] = run['round']
            return params
        else:
            inputResultsForRun = [result for result in loopResults if (result['$group'] == run['group'] and result['$round'] == run['input_round'])]
            inputResultsForRun = sorted(inputResultsForRun, key=lambda result: result['loss'])[0:run['configs_start']]

            existingResultsForRun = [result for result in loopResults if (result['$group'] == run['group'] and result['$round'] == run['round'])]

            inputCanonicalStrings = [json.dumps({key:value for key,value in result.items() if key not in OptimizationAlgorithmBase.resultInformationKeys}, sort_keys=True) for result in inputResultsForRun]
            existingCanonicalStrings = [json.dumps({key:value for key,value in result.items() if key not in OptimizationAlgorithmBase.resultInformationKeys}, sort_keys=True) for result in existingResultsForRun]

            neededCanonicalStrings = set(inputCanonicalStrings).difference(existingCanonicalStrings)
            neededResults = [inputResultsForRun[inputCanonicalStrings.index(resultString)] for resultString in neededCanonicalStrings]

            params = copy.deepcopy(random.choice(neededResults))

            params['$budget'] = run['budget']
            params['$loop'] = loop
            params['$group'] = run['group']
            params['$round'] = run['round']

            for key in OptimizationAlgorithmBase.resultInformationKeys:
                del params[key]

            return params
