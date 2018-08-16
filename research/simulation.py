import hyperopt
import math
import json
import random
import numpy
import functools
import concurrent.futures
import os
import sys
import time
import datetime
import subprocess
import scipy.stats
import csv
from scipy.stats import norm
import scipy.interpolate
from hypermax.utils import roundPrecision
from hypermax.hyperparameter import Hyperparameter
from pprint import pprint
import lightgbm as lgb

default_max_workers = 200

class AlgorithmSimulation:
    """ This class represents a simulation of hypothetical machine learning algorithm hyper-parameter spaces.

        It is mostly used for conducting abstract research into hyper-parameter optimization.
    """

    def __init__(self, independentInteractions=False):
        self.parameterCount = 0
        self.parameters = []
        self.computeScript = None
        self.search = None
        self.interactionCounts = {}
        for n in range(4):
            self.interactionCounts[n] = 0

        self.contributionCounts = {}
        for n in range(5):
            self.contributionCounts[n] = 0

        self.independentInteractions = independentInteractions

        self.createSearchFunction()

    def createHyperParameter(self):
        name = 'parameter_' + str(self.parameterCount)
        self.parameterCount += 1

        hasRounding = random.choice([True, False])
        if hasRounding:
            cardinality = random.randint(3,10)
        else:
            cardinality = 20

        weight = roundPrecision(random.uniform(0, 1))

        config = {
            "name": name,
            "cardinality": cardinality,
            "weight": roundPrecision(random.uniform(0, 1)),
            "space": {
                "type": "number",
                "min": 0,
                "max": 1,
                "scaling": "linear",
                "mode": "uniform",
                "weight": weight
            }
        }

        if hasRounding:
            config['space']['rounding'] = cardinality

        self.parameters.append(config)

        return config

    def createHyperParameterInteraction(self, param1, param2, type=None):
        if type is None:
            type = random.randint(0, 3)

        self.interactionCounts[type] += 1

        def square(weight):
            return weight * weight

        if type == 0:
            coords = [roundPrecision(random.uniform(0, 1)) for n in range(4)]

            coords[random.randint(0, 1)] = 0  # At least one of the four corners always touches 0, and one always touches 1
            coords[random.randint(2, 3)] = 1.0  # At least one of the four corners always touches 0, and one always touches 1

            xStart = coords[0]
            xEnd = coords[1]
            yStart = coords[2]
            yEnd = coords[3]

            xSlope = roundPrecision(xEnd - xStart)
            ySlope = roundPrecision(yEnd - yStart)

            maxVal = max(xStart, xEnd) * max(yStart, yEnd)

            return {
                "type": "linear",
                "func": "lambda x,y: ({0} + {1} * x) * ({2} + {3} * y) / {4}".format(xStart, xSlope, yStart, ySlope, maxVal),
                "param1": param1,
                "param2": param2,
                "weight": roundPrecision(square(random.uniform(0, 3)))
            }
        elif type == 1:
            peakX = roundPrecision(random.uniform(0, 1))
            peakY = roundPrecision(random.uniform(0, 1))
            spread = roundPrecision(random.uniform(0.5, 4.0))
            isHole = random.choice([True, False])

            if isHole:
                return {
                    "type": "peak",
                    "func": "lambda x, y: min(1.0, max(0, norm.pdf((x - {1}) * {0}) * norm.pdf((y - {3}) * {2}) * 7))".format(spread, peakX, spread, peakY),
                    "param1": param1,
                    "param2": param2,
                    "weight": roundPrecision(square(random.uniform(0, 3)))
                }
            else:
                return {
                    "type": "hole",
                    "func": "lambda x, y: min(1.0, max(0, 1.0 - norm.pdf((x - {1}) * {0}) * norm.pdf((y - {3}) * {2}) * 7))".format(spread, peakX, spread, peakY),
                    "param1": param1,
                    "param2": param2,
                    "weight": roundPrecision(square(random.uniform(0, 3)))
                }
        elif type == 2:
            xScale = roundPrecision(random.uniform(0.1, 3 * math.pi*2))
            yScale = roundPrecision(random.uniform(0.1, 3 * math.pi*2))

            xPhase = roundPrecision(random.uniform(0.1, 3 * math.pi*2))
            yPhase = roundPrecision(random.uniform(0.1, 3 * math.pi*2))

            return {
                "type": "waves",
                "func": "lambda x, y: (math.sin(x*{0} + {1}) + 1.0) * (math.sin(y*{2} + {3}) + 1.0) / 4.0".format(xScale, xPhase, yScale, yPhase),
                "param1": param1,
                "param2": param2,
                "weight": roundPrecision(square(random.uniform(0, 3)))
            }
        elif type == 3:
            sizeX = random.randint(3, 6)
            sizeY = random.randint(3, 6)
            grid = []
            for x in range(sizeY):
                row = []
                for y in range(sizeX):
                    row.append(roundPrecision(square(random.uniform(0, 1))))
                grid.append(row)

            return {
                "type": "random",
                "func": "scipy.interpolate.interp2d({0}, {1}, {2}, kind='linear')".format(json.dumps(list(numpy.linspace(0, 1.0, sizeX))),
                                                                                          json.dumps(list(numpy.linspace(0, 1.0, sizeY))), json.dumps(grid)),
                "param1": param1,
                "param2": param2,
                "weight": roundPrecision(square(random.uniform(0, 3)))
            }

    def createHyperParameterContribution(self, param, type=None):
        if type is None:
            type = random.randint(0, 4)

        self.contributionCounts[type] += 1

        def square(weight):
            return weight * weight

        if type == 0:
            xStart = roundPrecision(random.uniform(0, 1))
            xEnd = roundPrecision(random.uniform(0, 1))
            xSlope = roundPrecision(xEnd - xStart)

            return {
                "type": "linear",
                "func": "lambda x: ({0} + {1} * x)".format(xStart, xSlope),
                "param": param
            }
        elif type == 1:
            optimalPoint = roundPrecision(random.uniform(0, 1))

            invert = random.choice([True, False])

            if invert:
                return {
                    "type": "hill",
                    "func": "lambda x: min(1.0, max(0, 1.0 - ( math.sin(x*3.14 - {0}) / 2.0 + 0.5 ) ))".format(optimalPoint),
                    "param": param
                }
            else:
                return {
                    "type": "hill",
                    "func": "lambda x: min(1.0, max(0, ( math.sin(x*3.14 - {0}) / 2.0 + 0.5 )))".format(optimalPoint),
                    "param": param
                }
        elif type == 2:
            invert = random.choice([True, False])

            height = roundPrecision(random.uniform(0, 0.3))

            if invert:
                return {
                    "type": "exponential",
                    "func": "lambda x: min(1.0, max(0, 1.0 - (0.1 * math.pow(10, x) + {0})))".format(height),
                    "param": param
                }
            else:
                return {
                    "type": "exponential",
                    "func": "lambda x: min(1.0, max(0, 0.1 * (math.pow(10, x) + {0})))".format(height),
                    "param": param
                }
        elif type == 3:
            invert = random.choice([True, False])

            if invert:
                return {
                    "type": "logarithmic",
                    "func": "lambda x: min(1.0, max(0, 1.0 - (1.0 - math.log10(9*x+1))))",
                    "param": param
                }
            else:
                return {
                    "type": "logarithmic",
                    "func": "lambda x: min(1.0, max(0, (1.0 - math.log10(9*x+1))))",
                    "param": param
                }
        elif type == 4:
            # Random
            sizeX = random.randint(3, 8)
            values = [roundPrecision(random.uniform(0, 1)) for n in range(sizeX)]

            return {
                "type": "random",
                "func": "scipy.interpolate.interp1d({0}, {1})".format(json.dumps(list(numpy.linspace(0, 1, sizeX))), json.dumps(values)),
                "param": param
            }

    def createSearchFunction(self):
        parameters = [self.createHyperParameter() for n in range(random.randint(3, 16))]

        probabilityOfInteraction = 0.3

        contributions = []
        for parameter in parameters:
            contributions.append(self.createHyperParameterContribution(parameter))

        # We make two completely separate distributions. One for the first half of parameters, one for the second half.
        nameSortedParameters = sorted(parameter['name'] for parameter in parameters)
        cutoff = int(len(nameSortedParameters)/2)
        group1ParametersNames = nameSortedParameters[:cutoff]
        group2ParametersNames = nameSortedParameters[cutoff:]
        group1Parameters = [[p for p in parameters if p['name'] == parameter][0] for parameter in group1ParametersNames]
        group2Parameters = [[p for p in parameters if p['name'] == parameter][0] for parameter in group2ParametersNames]

        interactions = []
        if self.independentInteractions:
            for param1 in group1Parameters:
                for param2 in group1Parameters:
                    if param1['name'] != param2['name'] and random.uniform(0, 1) <= probabilityOfInteraction:
                        interactions.append(self.createHyperParameterInteraction(param1, param2))
            for param1 in group2Parameters:
                for param2 in group2Parameters:
                    if param1['name'] != param2['name'] and random.uniform(0, 1) <= probabilityOfInteraction:
                        interactions.append(self.createHyperParameterInteraction(param1, param2))
        else:
            for param1 in parameters:
                for param2 in parameters:
                    if param1['name'] != param2['name'] and random.uniform(0, 1) <= probabilityOfInteraction:
                        interactions.append(self.createHyperParameterInteraction(param1, param2))

        computeScript = ""
        computeScript += "from scipy.stats import norm\n"
        computeScript += "import math\n"
        computeScript += "import scipy.interpolate\n"
        computeScript += "\n"
        computeScript += "contributions = []\n"
        for contribution in contributions:
            computeScript += "contributions.append(" + contribution['func'] + ")\n"
        computeScript += "interactions = []\n"
        for interaction in interactions:
            computeScript += "interactions.append(" + interaction['func'] + ")\n"
        computeScript += "def computeLoss(params):\n"
        computeScript += "    loss = 0\n"
        totalParameterWeight = 0
        for parameterIndex, parameter in enumerate(parameters):
            computeScript += "    {0}_loss = 0\n".format(parameter['name'])
            computeScript += "    {0}_contribution = contributions[{1}](params[\"{2}\"])\n".format(parameter['name'], parameterIndex, parameter['name'])
            # computeScript += "    print(\"{0}_contribution\", {0}_contribution)\n".format(parameter['name'], parameter['name'])
            interactionsWeight = 0.0
            for index, interaction in enumerate(interactions):
                if interaction['param1']['name'] == parameter['name'] or interaction['param2']['name'] == parameter['name']:
                    computeScript += "    {0}_loss += interactions[{1}](params[\"{2}\"], params[\"{3}\"]) * {4}\n".format(parameter['name'], str(index),
                                                                                                                          interaction['param1']['name'],
                                                                                                                          interaction['param2']['name'], interaction['weight'])
                    interactionsWeight += interaction['weight']
                    # computeScript += "    print(\"interactions[{0}]\", interactions[{1}](params[\"{2}\"], params[\"{3}\"]) * {4})\n".format(str(index), str(index), interaction['param1']['name'], interaction['param2']['name'], interaction['weight'])
            contributionWeight = random.uniform(0.1, 0.4)
            computeScript += "    loss += {0}_loss * {1}\n".format(parameter['name'],
                                                                   parameter['weight'] / (interactionsWeight if interactionsWeight > 0 else 1.0) * (1.0 - contributionWeight))
            computeScript += "    loss += {0}_contribution * {1}\n".format(parameter['name'], parameter['weight'] * contributionWeight)
            totalParameterWeight += parameter['weight']

        computeScript += "    loss /= {0}\n".format(totalParameterWeight)
        # computeScript += "    print(loss)\n".format(totalParameterWeight)
        computeScript += "    return {\"loss\":float(loss[0]) if not isinstance(loss, float) else loss, \"status\": \"ok\"}\n"

        search = {
            "ui": {
                "enabled": False
            },
            "hyperparameters": {
                "type": "object",
                "properties": {param['name']: param['space'] for param in parameters},
            },
            "function": {
                "type": "python_function",
                "module": "test",
                "name": "computeLoss",
                "parallel": 25
            },
            "search": {
                "method": "random",
                "iterations": 10000
            },
            "results": {
                "graphs": True,
                "directory": "results"
            }
        }

        self.computeScript = computeScript
        self.search = search

    def run(self, gamma, processExecutor, lossFunction=None, initializationRounds=None, nEICandidates=None, priorWeight=None, secondaryCutoff=None, independentModellingRate=None):
        if self.computeScript is None:
            self.createSearchFunction()

        lossFutures = []
        # Run TPE a bunch of times with different lengths of trials
        for trialsLength in [100, 250]:
            for n in range(10):
                lossFutures.append(processExecutor.submit(runSearch, trialsLength, self.computeScript, gamma, self.search['hyperparameters'], lossFunction, initializationRounds, nEICandidates, priorWeight, secondaryCutoff, independentModellingRate))

        results = [future.result() for future in lossFutures]

        averages = {}

        # Compute the mean value for each of the statistics
        for stat in results[0].keys():
            values = [result[stat] for result in results]
            averages[stat] = numpy.mean(values)
        averages['gamma'] = gamma
        averages['lossFunction'] = lossFunction
        averages['initializationRounds'] = initializationRounds
        averages['nEICandidates'] = nEICandidates
        averages['priorWeight'] = priorWeight
        averages['secondaryCutoff'] = secondaryCutoff
        averages['independentModellingRate'] = independentModellingRate
        return averages



def convertResultsToTrials(parameterNames, results, lossFunc = None):
    trials = hyperopt.Trials()

    lossFunctions = {
        'identity': lambda x, results: x,
        'logarithmic': lambda x, results: math.log10(x) + 1.0,
        'squareroot': lambda x, results: math.sqrt(x),
        'exponential': lambda x, results: math.exp(x),
        'adapt5': functools.partial(adaptiveLossFunction, p=5),
        'adapt10': functools.partial(adaptiveLossFunction, p=10),
        'adapt25': functools.partial(adaptiveLossFunction, p=25),
        'adapt50': functools.partial(adaptiveLossFunction, p=50)
    }

    if lossFunc is None:
        lossFunc = 'identity'

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
            # 'result': {'loss': lossFunctions[lossFunc](result['loss'], results), 'status': 'ok'},
            'result': {'loss': result['loss'], 'status': 'ok'},
            'spec': None,
            'state': 2,
            'tid': resultIndex,
            'version': 0
        }

        for param in parameterNames:
            if param == 'activation' and isinstance(result[param], str): # Hack here just to get it working quickly
                values = ['relu', 'elu', "selu", "rrelu"]
                data['misc']['idxs'][param] = [resultIndex]
                data['misc']['vals'][param] = [values.index(result[param])]
            else:
                data['misc']['idxs'][param] = [resultIndex]
                data['misc']['vals'][param] = [result[param]]

        trials.insert_trial_doc(data)
    return trials

def adaptiveGamma(results, hyperparameters, n_startup_trials=20):
    if len(results) < n_startup_trials:
        return 1.0

    losses = numpy.array([result['loss'] for result in results])
    median = numpy.median(losses)
    standard_deviation = numpy.std(losses)
    skew = scipy.stats.skew(losses)

    log10_cardinality = 0.0
    for param in hyperparameters['properties']:
        if 'rounding' in hyperparameters['properties'][param]:
            log10_cardinality += math.log10((hyperparameters['properties'][param]['max'] - hyperparameters['properties'][param]['min']) / hyperparameters['properties'][param]['rounding'])
        else:
            log10_cardinality += math.log10(20)

    gamma = max(0.25, min(3.0, 0.23 + 0.53 * skew + log10_cardinality * 0.011 + (standard_deviation / median) * -0.3))
    return gamma

def adaptiveGammaGivenCardinality(results, log10_cardinality, n_startup_trials=20):
    if len(results) < n_startup_trials:
        return 1.0

    losses = numpy.array([result['loss'] for result in results])
    median = numpy.median(losses)
    standard_deviation = numpy.std(losses)
    skew = scipy.stats.skew(losses)

    gamma = max(0.25, min(3.0, 0.23 + 0.53 * skew + log10_cardinality * 0.011 + (standard_deviation / median) * -0.3))
    return gamma


def adaptiveNCandidates(hyperparameters):
    log10_cardinality = 0.0
    for param in hyperparameters['properties']:
        if 'rounding' in hyperparameters['properties'][param]:
            log10_cardinality += math.log10((hyperparameters['properties'][param]['max'] - hyperparameters['properties'][param]['min']) / hyperparameters['properties'][param]['rounding'])
        else:
            log10_cardinality += math.log10(20)

    n_EI_candidates = int(max(2, min(50, 1 + 40 * math.pow(10, -0.1 * log10_cardinality))))

    return n_EI_candidates

def adaptiveNCandidatesGivenCardinality(log10_cardinality):
    n_EI_candidates = int(max(2, min(50, 1 + 40 * math.pow(10, -0.1 * log10_cardinality))))
    return n_EI_candidates



def adaptiveSecondaryCutoff(results, hyperparameters):
    if len(results) < 50:
        return 0.0

    log10_cardinality = 0.0
    for param in hyperparameters['properties']:
        if 'rounding' in hyperparameters['properties'][param]:
            log10_cardinality += math.log10((hyperparameters['properties'][param]['max'] - hyperparameters['properties'][param]['min']) / hyperparameters['properties'][param]['rounding'])
        else:
            log10_cardinality += math.log10(20)

    losses = numpy.array([result['loss'] for result in results])
    median = numpy.median(losses)
    standard_deviation = numpy.std(losses)

    cutoff = max(0, min(1, 0.65 + (standard_deviation / median) * -0.07 + log10_cardinality * 0.02))

    return cutoff


def adaptiveSecondaryGivenCardinality(results, log10_cardinality):
    if len(results) < 50:
        return 0.0

    losses = numpy.array([result['loss'] for result in results])
    median = numpy.median(losses)
    standard_deviation = numpy.std(losses)

    cutoff = max(0, min(1, 0.65 + (standard_deviation / median) * -0.07 + log10_cardinality * 0.02))

    return cutoff


def runSearch(length, computeScript, gamma, hyperparameters, lossFunc=None, initializationRounds=None, nEICandidates=None, priorWeight=None, secondaryCutoff=None, independentModellingRate=None):
    testGlobals = {}
    exec(computeScript, testGlobals)
    computeLoss = testGlobals['computeLoss']

    losses = []

    best = None
    bestLoss = None

    if initializationRounds is None:
        initializationRounds = 20

    if nEICandidates is None:
        nEICandidates = 24

    if priorWeight is None:
        priorWeight = 1.0

    if secondaryCutoff is None:
        secondaryCutoff = 0.0

    currentResults = []
    def computePrimarySecondary():
        if len(currentResults) < initializationRounds:
            return list(sorted(hyperparameters['properties'].keys())), [] # Put all parameters as primary

        if len(set(result['loss'] for result in currentResults)) < 5:
            return list(sorted(hyperparameters['properties'].keys())), [] # Put all parameters as primary

        cutoffForTrial = secondaryCutoff
        if secondaryCutoff == 'auto':
            cutoffForTrial = adaptiveSecondaryCutoff(currentResults, hyperparameters)

        totalWeight = 0
        correlations = {}
        for parameter in hyperparameters['properties']:
            if len(set(result[parameter] for result in currentResults)) < 2:
                correlations[parameter] = 0
            else:
                correlation = abs(scipy.stats.spearmanr(numpy.array([result[parameter] for result in currentResults]), numpy.array([result['loss'] for result in currentResults]))[0])
                correlations[parameter] = correlation
                totalWeight += correlation

        threshold = totalWeight * cutoffForTrial

        primaryParameters = []
        secondaryParameters = []
        cumulative = totalWeight
        # Sort properties by their weight
        sortedParameters = sorted(hyperparameters['properties'].keys(), key=lambda parameter: -correlations[parameter])
        for parameter in sortedParameters:
            if cumulative < threshold:
                secondaryParameters.append(parameter)
            else:
                primaryParameters.append(parameter)

            cumulative -= correlations[parameter]
        return primaryParameters, secondaryParameters

    while len(currentResults) < length:
        next = {}
        def sample(parameters):
            nonlocal next
            next = parameters
            return {"loss": 0.5, 'status': 'ok'}

        space = {paramName: hyperopt.hp.uniform(paramName, 0, 1) for paramName in hyperparameters['properties']}

        primaryParameters, secondaryParameters = computePrimarySecondary()

        if best is not None:
            # Any secondary parameters have a 50-50 chance of being locked to their current best value.
            for secondary in secondaryParameters:
                if random.uniform(0, 1) < 0.5:
                    space[secondary] = best[secondary]

        willModelIndependently = False
        if independentModellingRate is not None:
            willModelIndependently = bool(random.uniform(0, 1.0) <= independentModellingRate)

        gammaForTrial = gamma
        if isinstance(gamma, str):
            n_startup_trials = int(gamma[gamma.find('_')+1:])
            gammaForTrial = adaptiveGamma(currentResults, hyperparameters, n_startup_trials=n_startup_trials)

        nEICandidatesForTrial = nEICandidates
        if isinstance(nEICandidates, str):
            nEICandidatesForTrial = adaptiveNCandidates(hyperparameters)

        if not willModelIndependently:
            hyperopt.fmin(fn=sample,
                          space=space,
                          algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=initializationRounds, gamma=gammaForTrial, n_EI_candidates=nEICandidatesForTrial, prior_weight=priorWeight),
                          max_evals=1,
                          trials=convertResultsToTrials(hyperparameters['properties'].keys(), currentResults, lossFunc=lossFunc),
                          rstate=numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1))))
        else:
            # We make two completely separate distributions. One for the first half of parameters, one for the second half.
            nameSortedParameters = sorted(hyperparameters['properties'].keys())
            cutoff = int(len(nameSortedParameters)/2)
            group1ParametersNames = nameSortedParameters[:cutoff]
            group2ParametersNames = nameSortedParameters[cutoff:]

            group1HyperoptSpace = {paramName: hyperoptConfig for paramName, hyperoptConfig in space.items() if paramName in group1ParametersNames}
            group2HyperoptSpace = {paramName: hyperoptConfig for paramName, hyperoptConfig in space.items() if paramName in group2ParametersNames}

            group1Hyperparameters = {
                "type": "object",
                "properties": {prop:config for prop,config in hyperparameters['properties'].items() if prop in group1ParametersNames}
            }

            group2Hyperparameters = {
                "type": "object",
                "properties": {prop:config for prop,config in hyperparameters['properties'].items() if prop in group2ParametersNames}
            }

            group1ResultsHistory = [
                {prop: value for prop, value in result.items() if (prop in group1ParametersNames or prop == 'loss' or prop == 'status')}
                for result in currentResults
            ]

            group2ResultsHistory = [
                {prop: value for prop, value in result.items() if (prop in group2ParametersNames or prop == 'loss' or prop == 'status')}
                for result in currentResults
            ]

            # Obtain the recommended hyper parameters for group 1
            hyperopt.fmin(fn=sample,
                          space=group1HyperoptSpace,
                          algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=initializationRounds, gamma=gamma, n_EI_candidates=nEICandidates, prior_weight=priorWeight),
                          max_evals=1,
                          trials=convertResultsToTrials(group1Hyperparameters, group1ResultsHistory, lossFunc=lossFunc),
                          rstate=numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1))))

            group1Next = dict(next)

            # Obtain the recommended hyper parameters for group 2
            hyperopt.fmin(fn=sample,
                          space=group2HyperoptSpace,
                          algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=initializationRounds, gamma=gamma, n_EI_candidates=nEICandidates, prior_weight=priorWeight),
                          max_evals=1,
                          trials=convertResultsToTrials(group2Hyperparameters, group2ResultsHistory, lossFunc=lossFunc),
                          rstate=numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1))))

            group2Next = dict(next)

            # Merge together the next recommended hyper-parameters for each group
            next = {}
            for key,value in group1Next.items():
                next[key]=value
            for key,value in group2Next.items():
                next[key]=value

        result = computeLoss(next)
        loss = result['loss']
        losses.append(loss)
        data = dict(next) # Clone the next dict
        data['loss'] = loss
        currentResults.append(data)

        if best is None or loss < bestLoss:
            best = data
            bestLoss = loss

    return {
        "loss": best['loss'],
        "skew": scipy.stats.skew(numpy.array(losses)),
        "std": scipy.std(losses),
        "kurtosis": scipy.stats.kurtosis(losses),
        "range": (numpy.min(losses), numpy.max(losses)),
        "min": numpy.min(losses),
        "median": numpy.median(losses),
        "max": numpy.max(losses),
        "best/median": (best['loss'] / numpy.median(losses))
    }


def createInteractionChartExample():
    algo = AlgorithmSimulation()
    param1 = algo.createHyperParameter()
    param2 = algo.createHyperParameter()
    interaction = algo.createHyperParameterInteraction(param1, param2, type=3)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    funcStore = {}
    exec("import math\nimport scipy.interpolate\nfrom scipy.stats import norm\nfunc = " + interaction['func'], funcStore)
    func = funcStore['func']

    xVals = numpy.linspace(0, 1, 25)
    yVals = numpy.linspace(0, 1, 25)

    grid = []
    for x in xVals:
        row = []
        for y in yVals:
            row.append(func(x, y)[0])
        grid.append(row)

    # Plot the surface.
    xVals, yVals = numpy.meshgrid(xVals, yVals)
    surf = ax.plot_surface(xVals, yVals, numpy.array(grid), cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=0, vmax=1)

    # Customize the z axis.
    ax.set_zlim(0, 1.00)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def createContributionChartExample():
    algo = AlgorithmSimulation()
    param1 = algo.createHyperParameter()
    contribution = algo.createHyperParameterContribution(param1, type=4)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib import cm

    fig, ax = plt.subplots()

    print(contribution['func'])
    funcStore = {}
    exec("import math\nimport scipy.interpolate\nfunc = " + contribution['func'], funcStore)
    func = funcStore['func']

    xVals = numpy.linspace(0, 1, 25)

    yVals = []
    for x in xVals:
        yVals.append(func(x))

    # Plot the surface.
    surf = ax.scatter(numpy.array(xVals), numpy.array(yVals), cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=0, vmax=1)

    plt.show()

lightGBMModel = None

def executeLightGBMModel(params, model=None):
    global lightGBMModel
    if model == 'textextraction':
        if lightGBMModel is None:
            lightGBMModel = lgb.Booster(model_file='LightGBM_model_text_extraction.txt')

        vectorKeys = [# They are in this order for a reason - thats what was in our training data file.
            'layer_0.max_depth',
            'layer_0.min_data_in_leaf',
            'layer_0.boosting_rounds',
            'layer_1.input_window',
            'layer_0.num_leaves',
            'layer_1.min_data_in_leaf',
            'layer_1.boosting_rounds',
            'layer_1.learning_rate',
            'layer_1.num_leaves',
            'layer_0.bagging_fraction',
            'layer_1.max_depth',
            'layer_0.learning_rate',
            'layer_0.input_window',
            'layer_0.feature_fraction']

        vector = []
        for param in vectorKeys:
            vector.append(params[param])

        result = lightGBMModel.predict([vector])[0]

        return {"loss": result, "status": "ok"}
    elif model == 'cifar_resnet':
        if lightGBMModel is None:
            lightGBMModel = lgb.Booster(model_file='LightGBM_model_cifar_resnet.txt')

        vectorKeys = [# They are in this order for a reason - thats what was in our training data file.
            'activation',
            'layer1_layers',
            'layer1_size',
            'layer2_layers',
            'layer2_size',
            'layer3_layers',
            'layer3_size',
            'layer4_layers',
            'layer4_size',
            'learning_rate',
            'weight_decay'
        ]

        vector = []
        for param in vectorKeys:
            if param == 'activation':
                values = ['relu', 'elu', "selu", "rrelu"]
                if isinstance(params[param], str):
                    vector.append(values.index(params[param]))
                else:
                    vector.append(params[param])
            else:
                vector.append(params[param])

        result = lightGBMModel.predict([vector])[0]

        return {"loss": result, "status": "ok"}


def optimizeLightGBMModel(gamma=None, nEICandidates=None, secondaryCutoff=None, initializationRounds=None, model=None):
    space = None
    log10_cardinality = None
    if model == 'textextraction':
        space = {
            "layer_0.max_depth": hyperopt.hp.quniform('layer_0.max_depth', 8, 16, 1),
            "layer_0.num_leaves": hyperopt.hp.quniform('layer_0.num_leaves', 5, 50, 1),
            "layer_0.min_data_in_leaf": hyperopt.hp.quniform('layer_0.min_data_in_leaf', 20, 80, 1),
            "layer_0.learning_rate": hyperopt.hp.loguniform('layer_0.learning_rate', 0.08, 0.5),
            "layer_0.boosting_rounds": hyperopt.hp.quniform('layer_0.boosting_rounds', 5, 75, 1),
            "layer_0.input_window": hyperopt.hp.quniform('layer_0.input_window', 0, 3, 1),
            "layer_0.feature_fraction": hyperopt.hp.uniform('layer_0.feature_fraction', 0.5, 1.0),
            "layer_0.bagging_fraction": hyperopt.hp.uniform('layer_0.bagging_fraction', 0.1, 1.0),
            "layer_1.max_depth": hyperopt.hp.quniform('layer_1.max_depth', 2, 16, 1),
            "layer_1.num_leaves": hyperopt.hp.quniform('layer_1.num_leaves', 5, 50, 1),
            "layer_1.min_data_in_leaf": hyperopt.hp.quniform('layer_1.min_data_in_leaf', 1, 100, 1),
            "layer_1.learning_rate": hyperopt.hp.loguniform('layer_1.learning_rate', 0.01, 0.3),
            "layer_1.boosting_rounds": hyperopt.hp.quniform('layer_1.boosting_rounds', 5, 75, 1),
            "layer_1.input_window": hyperopt.hp.quniform('layer_1.input_window', 3, 5, 1)
        }
        log10_cardinality = 16.013760489
    elif model == 'cifar_resnet':
        space = {
            "learning_rate": hyperopt.hp.uniform('learning_rate', 0.1, 3.0),
            "weight_decay": hyperopt.hp.loguniform('weight_decay', math.log(2e-5), math.log(2e-3)),
            "activation": hyperopt.hp.choice('activation', ['relu', 'elu', "selu", "rrelu"]),
            "layer1_size": hyperopt.hp.quniform('layer1_size', 32, 96, 16),
            "layer1_layers": hyperopt.hp.quniform('layer1_layers', 1, 3, 1),
            "layer2_size": hyperopt.hp.quniform('layer2_size', 64, 192, 32),
            "layer2_layers": hyperopt.hp.quniform('layer2_layers', 1, 3, 1),
            "layer3_size": hyperopt.hp.quniform('layer3_size', 128, 384, 64),
            "layer3_layers": hyperopt.hp.quniform('layer3_layers', 1, 3, 1),
            "layer4_size": hyperopt.hp.quniform('layer4_size', 256, 768, 128),
            "layer4_layers": hyperopt.hp.quniform('layer4_layers', 1, 3, 1),
        }
        log10_cardinality = 7.908485019

    losses = []
    for length in [100, 250]:
        for n in range(10):
            random.seed(time.time())
            if nEICandidates is None:
                nEICandidates = 24

            if gamma is None:
                gamma = 0.25

            if secondaryCutoff is None:
                secondaryCutoff = 0.0

            if initializationRounds is None:
                initializationRounds = 20

            currentResults = []

            def computePrimarySecondary():
                if len(currentResults) < 20:
                    return list(sorted(space.keys())), []  # Put all parameters as primary

                if len(set('{:.3E}'.format(result['loss']) for result in currentResults)) < 5:
                    return list(sorted(space.keys())), []  # Put all parameters as primary

                cutoffForTrial = secondaryCutoff
                if secondaryCutoff == 'auto':
                    cutoffForTrial = adaptiveSecondaryGivenCardinality(currentResults, log10_cardinality)

                totalWeight = 0
                correlations = {}
                for parameter in space.keys():
                    if len(set(result[parameter] for result in currentResults)) < 2:
                        correlations[parameter] = 0
                    else:
                        correlation = abs(scipy.stats.spearmanr(numpy.array([result[parameter] for result in currentResults]), numpy.array([result['loss'] for result in currentResults]))[0])
                        correlations[parameter] = correlation
                        totalWeight += correlation

                threshold = totalWeight * cutoffForTrial

                primaryParameters = []
                secondaryParameters = []
                cumulative = totalWeight
                # Sort properties by their weight
                sortedParameters = sorted(space.keys(), key=lambda parameter: -correlations[parameter])
                for parameter in sortedParameters:
                    if cumulative < threshold:
                        secondaryParameters.append(parameter)
                    else:
                        primaryParameters.append(parameter)

                    cumulative -= correlations[parameter]
                return primaryParameters, secondaryParameters

            best = None
            bestLoss = None

            while len(currentResults) < length:
                next = {}

                def sample(parameters):
                    nonlocal next
                    next = parameters
                    return {"loss": 0.5, 'status': 'ok'}

                primaryParameters, secondaryParameters = computePrimarySecondary()

                spaceForTrial = {key: param for key,param in space.items()}

                if best is not None:
                    # Any secondary parameters have a 50-50 chance of being locked to their current best value.
                    for secondary in secondaryParameters:
                        if random.uniform(0, 1) < 0.5:
                            spaceForTrial[secondary] = best[secondary]

                gammaForTrial = gamma
                if isinstance(gamma, str):
                    n_startup_trials = int(gamma[gamma.find('_') + 1:])
                    gammaForTrial = adaptiveGammaGivenCardinality(currentResults, log10_cardinality=log10_cardinality, n_startup_trials=n_startup_trials)

                nEICandidatesForTrial = nEICandidates
                if isinstance(nEICandidates, str):
                    nEICandidatesForTrial = adaptiveNCandidatesGivenCardinality(log10_cardinality=log10_cardinality)

                hyperopt.fmin(fn=sample,
                              space=spaceForTrial,
                              algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=initializationRounds, gamma=gammaForTrial, n_EI_candidates=nEICandidatesForTrial),
                              max_evals=1,
                              trials=convertResultsToTrials(space.keys(), currentResults, lossFunc='identity'),
                              rstate=numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1))))


                result = executeLightGBMModel(next, model)

                loss = result['loss']
                data = dict(next)  # Clone the next dict
                data['loss'] = loss
                currentResults.append(data)

                if best is None or loss < bestLoss:
                    best = data
                    bestLoss = loss

            losses.append(bestLoss)
            sys.stdout.flush()

    return numpy.mean(losses)

def addAlgorithmStatistics(algorithm, data):
    data['num_parameters'] = len(algorithm.parameters)

    data['log10_cardinality'] = 0.0
    for param in algorithm.parameters:
        data['log10_cardinality'] += math.log10(float(param['cardinality']))

    data['interactions'] = float(algorithm.interactionCounts[0] + algorithm.interactionCounts[1] + algorithm.interactionCounts[2] + algorithm.interactionCounts[3])
    data['interactions_linear'] = float(algorithm.interactionCounts[0]) / max(1, data['interactions'])
    data['interactions_peakvalley'] = float(algorithm.interactionCounts[1]) / max(1, data['interactions'])
    data['interactions_wave'] = float(algorithm.interactionCounts[2]) / max(1, data['interactions'])
    data['interactions_random'] = float(algorithm.interactionCounts[3]) / max(1, data['interactions'])
    data['interactions_index'] = float(max(1, data['interactions'])) / float(data['num_parameters'] * data['num_parameters'] * 0.3)

    data['contributions_linear'] = float(algorithm.contributionCounts[0]) / data['num_parameters']
    data['contributions_peakvalley'] = float(algorithm.contributionCounts[1]) / data['num_parameters']
    data['contributions_exponential'] = float(algorithm.contributionCounts[2]) / data['num_parameters']
    data['contributions_logarithmic'] = float(algorithm.contributionCounts[3]) / data['num_parameters']
    data['contributions_random'] = float(algorithm.contributionCounts[4]) / data['num_parameters']

def testGammaStatistics():
    algorithms = [AlgorithmSimulation() for n in range(500)]

    gammaValues = [0.25, 1.0, 'adaptive_20', 'adaptive_40', 'adaptive_60', 'adaptive_80']

    results = []
    for algorithm in algorithms:
        # Find the optimal gamma value for each algorithm
        start = datetime.datetime.now()
        with concurrent.futures.ProcessPoolExecutor(max_workers=default_max_workers) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
                resultFutures = []
                for gamma in gammaValues:
                    resultFutures.append(threadExecutor.submit(lambda algorithm, gamma, processExecutor: algorithm.run(gamma, processExecutor), algorithm, gamma, processExecutor))

        algorithmResults = {future.result()['gamma']: future.result() for future in resultFutures}

        data = algorithmResults[1.0]
        for gamma in gammaValues:
            data['gamma_'+str(gamma)] = algorithmResults[gamma]['loss']
        del data['gamma']

        results.append(data)

        with open('results.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(data.keys()))
            writer.writeheader()
            writer.writerows(results)

        pprint(data)

def adaptiveLossFunction(x, results, p):
    if len(results) < 20: # Only start doing adaptive loss after 20 rounds.
        return x

    best = numpy.min([result['loss'] for result in results])
    percentile = numpy.percentile([result['loss'] for result in results], p)

    if best == percentile: # the math function is undefined if this is the case
        return x

    cost = ((x - best) * (0.99 / (percentile - best)) + 1.01) * math.log10((x - best) * (0.99 / (percentile - best)) + 0.01) + 1
    return cost


def testLossFunctions():
    algorithms = [AlgorithmSimulation() for n in range(1000)]

    lossFunctions = [
        'identity',
        'logarithmic',
        'squareroot',
        'exponential',
        'adapt5',
        'adapt10',
        'adapt25',
        'adapt50'
    ]

    gamma = 1.0

    results = []
    for algorithm in algorithms:
        # Find the optimal gamma value for each algorithm
        with concurrent.futures.ProcessPoolExecutor(max_workers=default_max_workers) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
                resultFutures = []
                for lossFunction in lossFunctions:
                    resultFutures.append(threadExecutor.submit(lambda algorithm, gamma, lossFunction, processExecutor: algorithm.run(gamma, processExecutor, lossFunction=lossFunction), algorithm, gamma, lossFunction, processExecutor))

        algorithmResults = {future.result()['lossFunction']: future.result() for future in resultFutures}
        data = algorithmResults['identity']
        data['identity'] = data['loss']

        for name in lossFunctions:
            data[name] = algorithmResults[name]['loss']

        del data['loss']
        del data['lossFunction']
        addAlgorithmStatistics(algorithm, data)
        results.append(data)

        with open('results.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(data.keys()))
            writer.writeheader()
            writer.writerows(results)

        print("Latest!")
        pprint(data)



def testInitializationRounds():
    algorithms = [AlgorithmSimulation() for n in range(500)]

    initializationRoundsToTest = [
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90
    ]

    gamma = 1.0

    results = []
    for algorithm in algorithms:
        # Find the optimal gamma value for each algorithm
        with concurrent.futures.ProcessPoolExecutor(max_workers=default_max_workers) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
                resultFutures = []
                for rounds in initializationRoundsToTest:
                    resultFutures.append(threadExecutor.submit(lambda algorithm, gamma, initializationRounds, processExecutor: algorithm.run(gamma, processExecutor, initializationRounds=initializationRounds), algorithm, gamma, rounds, processExecutor))

        algorithmResults = {future.result()['initializationRounds']: future.result() for future in resultFutures}

        data = algorithmResults[20]
        data['initialization_rounds_20'] = data['loss']

        for rounds in initializationRoundsToTest:
            data['initialization_rounds_' + str(rounds)] = algorithmResults[rounds]['loss']

        del data['loss']
        del data['initializationRounds']
        addAlgorithmStatistics(algorithm, data)
        results.append(data)

        with open('results.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(data.keys()))
            writer.writeheader()
            writer.writerows(results)

        print("Latest!")
        pprint(data)




def testNEICandidates():
    algorithms = [AlgorithmSimulation() for n in range(500)]

    nEICandidatesToTest = [
        24,
        'adaptive'
    ]

    gamma = 1.0

    results = []
    for algorithm in algorithms:
        # Find the optimal gamma value for each algorithm
        with concurrent.futures.ProcessPoolExecutor(max_workers=default_max_workers) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
                resultFutures = []
                for nEICandidates in nEICandidatesToTest:
                    resultFutures.append(threadExecutor.submit(lambda algorithm, gamma, nEICandidates, processExecutor: algorithm.run(gamma, processExecutor, nEICandidates=nEICandidates), algorithm, gamma, nEICandidates, processExecutor))

        algorithmResults = {future.result()['nEICandidates']: future.result() for future in resultFutures}

        data = algorithmResults[24]
        data['n_ei_candidates_24'] = data['loss']

        for rounds in nEICandidatesToTest:
            data['n_ei_candidates_' + str(rounds)] = algorithmResults[rounds]['loss']

        del data['loss']
        del data['nEICandidates']
        addAlgorithmStatistics(algorithm, data)
        results.append(data)

        with open('results.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(data.keys()))
            writer.writeheader()
            writer.writerows(results)

        print("Latest!")
        pprint(data)


def testPriorWeight():
    algorithms = [AlgorithmSimulation() for n in range(500)]

    priorWeightsToTest = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5
    ]

    gamma = 1.0

    results = []
    for algorithm in algorithms:
        # Find the optimal gamma value for each algorithm
        with concurrent.futures.ProcessPoolExecutor(max_workers=default_max_workers) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
                resultFutures = []
                for priorWeight in priorWeightsToTest:
                    resultFutures.append(threadExecutor.submit(lambda algorithm, gamma, priorWeight, processExecutor: algorithm.run(gamma, processExecutor, priorWeight=priorWeight), algorithm, gamma, priorWeight, processExecutor))

        algorithmResults = {future.result()['priorWeight']: future.result() for future in resultFutures}

        data = algorithmResults[1.0]
        data['n_prior_weight_1.0'] = data['loss']

        for priorWeight in priorWeightsToTest:
            data['n_prior_weight_' + str(priorWeight)] = algorithmResults[priorWeight]['loss']

        del data['loss']
        del data['priorWeight']
        addAlgorithmStatistics(algorithm, data)
        results.append(data)

        with open('results.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(data.keys()))
            writer.writeheader()
            writer.writerows(results)

        print("Latest!")
        pprint(data)

def testHyperParameterWeights():
    algorithms = [AlgorithmSimulation() for n in range(500)]

    # secondaryGroupCutoffs = [
    #     0.0,
    #     0.1,
    #     0.2,
    #     0.3,
    #     0.4,
    #     0.5,
    #     0.6,
    #     0.7,
    #     0.8,
    #     0.9,
    #     1.0
    # ]

    secondaryGroupCutoffs = [
        0.0,
        'auto'
    ]

    gamma = 1.0

    results = []
    for algorithm in algorithms:
        # Find the optimal gamma value for each algorithm
        with concurrent.futures.ProcessPoolExecutor(max_workers=default_max_workers) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
                resultFutures = []
                for secondaryCutoff in secondaryGroupCutoffs:
                    resultFutures.append(threadExecutor.submit(lambda algorithm, gamma, secondaryCutoff, processExecutor: algorithm.run(gamma, processExecutor, secondaryCutoff=secondaryCutoff), algorithm, gamma, secondaryCutoff, processExecutor))

        algorithmResults = {future.result()['secondaryCutoff']: future.result() for future in resultFutures}

        data = algorithmResults[0.0]
        data['secondaryCutoff_0.0'] = data['loss']

        for secondaryCutoff in secondaryGroupCutoffs:
            data['secondaryCutoff_' + str(secondaryCutoff)] = algorithmResults[secondaryCutoff]['loss']

        del data['loss']
        del data['secondaryCutoff']
        addAlgorithmStatistics(algorithm, data)
        results.append(data)

        with open('results.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(data.keys()))
            writer.writeheader()
            writer.writerows(results)

        print("Latest!")
        pprint(data)



def testIndependentDistributionModelling():
    algorithms = [AlgorithmSimulation(independentInteractions=True) for n in range(500)]

    independentDistributionModellingRates = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ]

    gamma = 1.0

    results = []
    for algorithm in algorithms:
        # Find the optimal gamma value for each algorithm
        with concurrent.futures.ProcessPoolExecutor(max_workers=default_max_workers) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
                resultFutures = []
                for independentModellingRate in independentDistributionModellingRates:
                    resultFutures.append(threadExecutor.submit(lambda algorithm, gamma, independentModellingRate, processExecutor: algorithm.run(gamma, processExecutor, independentModellingRate=independentModellingRate), algorithm, gamma, independentModellingRate, processExecutor))

        algorithmResults = {future.result()['independentModellingRate']: future.result() for future in resultFutures}

        data = algorithmResults[0.0]
        data['independentModellingRate_0.0'] = data['loss']

        for independentModellingRate in independentDistributionModellingRates:
            data['independentModellingRate_' + str(independentModellingRate)] = algorithmResults[independentModellingRate]['loss']

        del data['loss']
        del data['independentModellingRate']
        addAlgorithmStatistics(algorithm, data)
        results.append(data)

        with open('results.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(data.keys()))
            writer.writeheader()
            writer.writerows(results)

        print("Latest!")
        pprint(data)




if __name__ == '__main__':
    gammaValues = [0.25, 1.0, 'adaptive_20', 'adaptive_40', 'adaptive_60', 'adaptive_80']
    # gammaValues = ['adaptive_40', 'adaptive_60', 'adaptive_80']
    # initializationRounds = [5,10,20,40]

    # n_EI_candidates = [24, 'auto']

    for model in ['textextraction']:
        # secondaryGroupCutoffs = [
        #     0.0,
        #     0.9,
        #     'auto'
        # ]

        # lightGBMModel /= None

        # results = {}
        # for value in gammaValues:

        # print("Testing TPE")
        # sys.stdout.flush()
        # start = datetime.datetime.now()
        # loss = optimizeLightGBMModel(gamma=0.25, nEICandidates=24, secondaryCutoff=0.0, initializationRounds=20, model=model)
        # end = datetime.datetime.now()
        # print((end-start).total_seconds(), "(s)   Loss: ", loss)
        # sys.stdout.flush()

        print("Testing ABTPE")
        sys.stdout.flush()
        start = datetime.datetime.now()
        loss = optimizeLightGBMModel(gamma=0.25, nEICandidates='auto', secondaryCutoff='auto', initializationRounds=10, model=model)
        end = datetime.datetime.now()
        print((end - start).total_seconds(), "(s)   Loss: ", loss)
        sys.stdout.flush()

    # createInteractionChartExample()

    # testGammaStatistics()
    #
    # testInitializationRounds()

    # testLossFunctions()

    # testNEICandidates()

    # testPriorWeight()

    # testHyperParameterWeights()
    # testIndependentDistributionModelling()

# reset && rsync -racv ~/hypermax/ 40.87.46.41:hypermax && ssh 40.87.46.41 "killall python3; source venv/bin/activate && cd hypermax && python hypermax/simulation.py && cd .."
