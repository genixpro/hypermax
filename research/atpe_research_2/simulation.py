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
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.interpolate
import copy
import sklearn.preprocessing
import sklearn.cluster
from hypermax.utils import roundPrecision
from hypermax.hyperparameter import Hyperparameter
from pprint import pprint
import psutil
import lightgbm as lgb

default_max_workers = int(psutil.cpu_count()*1.1)

class AlgorithmSimulation:
    """ This class represents a simulation of hypothetical machine learning algorithm hyper-parameter spaces.

        It is mostly used for conducting abstract research into hyper-parameter optimization.
    """

    def __init__(self):
        self.parameterCount = 0
        self.parameters = []
        self.log10_cardinality = None
        self.computeScript = None
        self.computeLoss = None
        self.search = None
        self.interactionCounts = {}
        self.interactionTypes = ['linear', 'peakvalley', 'wave', 'random']
        for n in self.interactionTypes:
            self.interactionCounts[n] = 0

        self.contributionCounts = {}
        self.contributionTypes = ['linear', 'peakvalley', 'exponential', 'logarithmic', 'random']
        for n in self.contributionTypes:
            self.contributionCounts[n] = 0

        self.noiseFactor = random.uniform(1.01, 1.15)
        self.failRate = random.uniform(0.0, 0.1)

        self.createSearchFunction()

    def createHyperParameter(self, group):
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
            "weight": weight,
            "group": group,
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
            config['space']['rounding'] = 1.0 / cardinality
            config['rounding'] = 1.0 / cardinality

        self.parameters.append(config)

        return config

    def createHyperParameterInteraction(self, param1, param2, group, type=None):
        if type is None:
            type = random.choice(self.interactionTypes + ['random']) # Increase weight of random interactions

        self.interactionCounts[type] += 1

        def square(weight):
            return weight * weight

        if type == 'linear':
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
                "weight": roundPrecision(square(random.uniform(0, 3))),
                "group": group
            }
        elif type == 'peakvalley':
            peakX = roundPrecision(random.uniform(0, 1))
            peakY = roundPrecision(random.uniform(0, 1))
            spread = roundPrecision(random.uniform(0.5, 4.0))
            isHole = random.choice([True, False])

            if isHole:
                return {
                    "type": "peakvalley",
                    "func": "lambda x, y: min(1.0, max(0, norm.pdf((x - {1}) * {0}) * norm.pdf((y - {3}) * {2}) * 7))".format(spread, peakX, spread, peakY),
                    "param1": param1,
                    "param2": param2,
                    "weight": roundPrecision(square(random.uniform(0, 3))),
                    "group": group
                }
            else:
                return {
                    "type": "peakvalley",
                    "func": "lambda x, y: min(1.0, max(0, 1.0 - norm.pdf((x - {1}) * {0}) * norm.pdf((y - {3}) * {2}) * 7))".format(spread, peakX, spread, peakY),
                    "param1": param1,
                    "param2": param2,
                    "weight": roundPrecision(square(random.uniform(0, 3))),
                    "group": group
                }
        elif type == 'wave':
            xScale = roundPrecision(random.uniform(0.1, 3 * math.pi*2))
            yScale = roundPrecision(random.uniform(0.1, 3 * math.pi*2))

            xPhase = roundPrecision(random.uniform(0.1, 3 * math.pi*2))
            yPhase = roundPrecision(random.uniform(0.1, 3 * math.pi*2))

            return {
                "type": "wave",
                "func": "lambda x, y: (math.sin(x*{0} + {1}) + 1.0) * (math.sin(y*{2} + {3}) + 1.0) / 4.0".format(xScale, xPhase, yScale, yPhase),
                "param1": param1,
                "param2": param2,
                "weight": roundPrecision(square(random.uniform(0, 3))),
                "group": group
            }
        elif type == 'random':
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
                "func": "scipy.interpolate.interp2d({0}, {1}, {2}, kind='linear')".format(json.dumps([roundPrecision(n) for n in numpy.linspace(0, 1.0, sizeX)]),
                                                                                          json.dumps([roundPrecision(n) for n in numpy.linspace(0, 1.0, sizeY)]),
                                                                                          json.dumps(grid)),
                "param1": param1,
                "param2": param2,
                "weight": roundPrecision(square(random.uniform(0, 3))),
                "group": group
            }

    def createHyperParameterContribution(self, param, group, type=None):
        if type is None:
            type = random.choice(self.contributionTypes + ['random']) # Increase weight of random interactions

        self.contributionCounts[type] += 1

        def square(weight):
            return weight * weight

        if type == 'linear':
            xStart = roundPrecision(random.uniform(0, 1))
            xEnd = roundPrecision(random.uniform(0, 1))
            xSlope = roundPrecision(xEnd - xStart)

            return {
                "type": "linear",
                "func": "lambda x: ({0} + {1} * x)".format(xStart, xSlope),
                "param": param,
                "group": group
            }
        elif type == 'peakvalley':
            optimalPoint = roundPrecision(random.uniform(0, 1))

            invert = random.choice([True, False])

            if invert:
                return {
                    "type": "peakvalley",
                    "func": "lambda x: min(1.0, max(0, 1.0 - ( math.sin(x*3.14 - {0}) / 2.0 + 0.5 ) ))".format(optimalPoint),
                    "param": param,
                    "group": group
                }
            else:
                return {
                    "type": "peakvalley",
                    "func": "lambda x: min(1.0, max(0, ( math.sin(x*3.14 - {0}) / 2.0 + 0.5 )))".format(optimalPoint),
                    "param": param,
                    "group": group
                }
        elif type == 'exponential':
            invertX = random.choice([True, False])
            invertY = random.choice([True, False])

            height = roundPrecision(random.uniform(0, 0.3))

            steepNess = roundPrecision(random.uniform(2, 30) ** 3)

            if invertY:
                if invertX:
                    return {
                        "type": "exponential",
                        "func": "lambda x: min(1.0, max(0, 1.0 - ({0} * math.pow({1}, (1.0 - x)) + {2})))".format(1.0/steepNess, steepNess, height),
                        "param": param,
                        "group": group
                    }
                else:
                    return {
                        "type": "exponential",
                        "func": "lambda x: min(1.0, max(0, 1.0 - ({0} * math.pow({1}, x) + {2})))".format(1.0/steepNess, steepNess, height),
                        "param": param,
                        "group": group
                    }
            else:
                if invertX:
                    return {
                        "type": "exponential",
                        "func": "lambda x: min(1.0, max(0, {0} * (math.pow({1}, (1.0 - x)) + {2})))".format(1.0/steepNess, steepNess, height),
                        "param": param,
                        "group": group
                    }
                else:
                    return {
                        "type": "exponential",
                        "func": "lambda x: min(1.0, max(0, {0} * (math.pow({1}, x) + {2})))".format(1.0 / steepNess, steepNess, height),
                        "param": param,
                        "group": group
                    }
        elif type == 'logarithmic':
            invertX = random.choice([True, False])
            invertY = random.choice([True, False])

            steepNess = roundPrecision(random.uniform(3, 30) ** 2)

            if invertY:
                if invertX:
                    return {
                        "type": "logarithmic",
                        "func": "lambda x: min(1.0, max(0, 1.0 - (1.0 - math.log({0}*(1.0-x)+1, {1}))))".format(steepNess-1, steepNess),
                        "param": param,
                        "group": group
                    }
                else:
                    return {
                        "type": "logarithmic",
                        "func": "lambda x: min(1.0, max(0, 1.0 - (1.0 - math.log({0}*x+1, {1}))))".format(steepNess-1, steepNess),
                        "param": param,
                        "group": group
                    }
            else:
                if invertX:
                    return {
                        "type": "logarithmic",
                        "func": "lambda x: min(1.0, max(0, (1.0 - math.log({0}*(1.0-x)+1, {1}))))".format(steepNess-1, steepNess),
                        "param": param,
                        "group": group
                    }
                else:
                    return {
                        "type": "logarithmic",
                        "func": "lambda x: min(1.0, max(0, (1.0 - math.log({0}*x+1, {1}))))".format(steepNess - 1, steepNess),
                        "param": param,
                        "group": group
                    }
        elif type == 'random':
            # Random
            sizeX = random.randint(4, 9)
            values = [roundPrecision(random.uniform(0, 1)) for n in range(sizeX)]

            return {
                "type": "random",
                "func": "scipy.interpolate.interp1d({0}, {1})".format(json.dumps([roundPrecision(n) for n in numpy.linspace(0, 1, sizeX)]), json.dumps(values)),
                "param": param,
                "group": group
            }


    def createSearchFunction(self):
        primaryParameters = [self.createHyperParameter(group='primary') for n in range(random.randint(3, 10))]

        parameterSubGroups = random.randint(1, 3)
        probabilityOfInteraction = random.uniform(0.1, 0.5)

        contributions = []
        for parameter in primaryParameters:
            contributions.append(self.createHyperParameterContribution(parameter, group='primary'))

        interactions = []
        for param1Index, param1 in enumerate(primaryParameters):
            for param2Index, param2 in enumerate(primaryParameters[param1Index+1:]):
                if param1['name'] != param2['name'] and random.uniform(0, 1) <= probabilityOfInteraction:
                    interactions.append(self.createHyperParameterInteraction(param1, param2, group='primary'))

        allContributions = list(contributions)
        allInteractions = list(interactions)

        subGroups = []
        for n in range(parameterSubGroups):
            group = 'group' + str(n+1)
            subGroupParameters = [self.createHyperParameter(group=group) for n in range(random.randint(2, 8))]
            subGroupContributions = [self.createHyperParameterContribution(parameter, group=group) for parameter in subGroupParameters]

            parametersForInteraction = primaryParameters + subGroupParameters

            subGroupInteractions = []
            for param1Index, param1 in enumerate(parametersForInteraction):
                for param2Index, param2 in enumerate(parametersForInteraction[param1Index+1:]):
                    if param1['name'] != param2['name'] and random.uniform(0, 1) <= probabilityOfInteraction and (param1 in subGroupParameters or param2 in subGroupParameters):
                        subGroupInteractions.append(self.createHyperParameterInteraction(param1, param2, group=group))
            subGroups.append({
                "parameters": subGroupParameters,
                "contributions": subGroupContributions,
                "interactions": subGroupInteractions
            })

            allContributions = allContributions + subGroupContributions
            allInteractions = allInteractions + subGroupInteractions


        computeScript = ""
        computeScript += "from scipy.stats import norm\n"
        computeScript += "import math\n"
        computeScript += "import random\n"
        computeScript += "import scipy.interpolate\n"
        computeScript += "\n"
        computeScript += "contributions = []\n"
        for index, contribution in enumerate(allContributions):
            computeScript += "contributions.append(" + contribution['func'] + ")\n"
            contribution['index'] = index
        computeScript += "interactions = []\n"
        for index, interaction in enumerate(allInteractions):
            computeScript += "interactions.append(" + interaction['func'] + ")\n"
            interaction['index'] = index
        computeScript += "def computeLoss(params):\n"
        computeScript += "    #print(params)\n"
        computeScript += "    if random.uniform(0, 1.0) < {0}:\n".format(self.failRate)
        computeScript += "        return {\"loss\": 1.0, \"status\": \"ok\"}\n"

        def getParamFetcher(param):
            if param['group'] == 'primary':
                return "params[\"{0}\"]".format(param['name'])
            else:
                return "params[\"group\"][\"{1}\"]".format(param['group'], param['name'])

        for index, subGroup in enumerate(subGroups):
            # Decide which sub-group is getting optimized
            computeScript += "    if params['group']['group'] == 'group{0}':\n".format(index+1)
            computeScript += "        loss = 0\n"
            totalParameterWeight = 0
            group = 'group' + str(index+1)

            contributionsForGroup = [contribution for contribution in allContributions if contribution['group'] == 'primary' or contribution['group'] == group]
            interactionsForGroup = [interaction for interaction in allInteractions if interaction['group'] == 'primary' or interaction['group'] == group]
            parametersForGroup = [(contribution['param'], contribution['index']) for contribution in contributionsForGroup]

            for parameterIndex, parameter in enumerate(parametersForGroup):
                computeScript += "        {0}_loss = 0\n".format(parameter[0]['name'])
                computeScript += "        {0}_contribution = contributions[{1}]({2})\n".format(parameter[0]['name'], parameter[1], getParamFetcher(parameter[0]))

                interactionsWeight = 0.0
                for index, interaction in enumerate(interactionsForGroup):
                    if interaction['param1']['name'] == parameter[0]['name'] or interaction['param2']['name'] == parameter[0]['name']:
                        computeScript += "        {0}_loss += interactions[{1}]({2}, {3} * {4})\n".format(parameter[0]['name'], str(index),
                                                                                                                              getParamFetcher(interaction['param1']),
                                                                                                                              getParamFetcher(interaction['param2']), interaction['weight'])
                        interactionsWeight += interaction['weight']

                contributionWeight = random.uniform(0.1, 0.4)
                computeScript += "        loss += {0}_loss * {1}\n".format(parameter[0]['name'],
                                                                       parameter[0]['weight'] / (interactionsWeight if interactionsWeight > 0 else 1.0) * (1.0 - contributionWeight))
                computeScript += "        loss += {0}_contribution * {1}\n".format(parameter[0]['name'], parameter[0]['weight'] * contributionWeight)
                totalParameterWeight += parameter[0]['weight']

            computeScript += "        loss /= {0}\n".format(totalParameterWeight)
            # computeScript += "        print(loss)\n".format(totalParameterWeight)
            computeScript += "        loss *= random.uniform(1.0, {0})\n".format(self.noiseFactor)
            computeScript += "        return {\"loss\":float(loss[0]) if not isinstance(loss, float) else loss, \"status\": \"ok\"}\n"

        groups = [{
            'type': 'object',
            'properties': {param['name']: param['space'] for param in group['parameters']}
        } for group in subGroups]


        parameterSpace = {param['name']: param['space'] for param in primaryParameters}
        for index, group in enumerate(groups):
            parameterSpace['group'+str(index+1)] = group

        search = {
            "ui": {
                "enabled": False
            },
            "hyperparameters": {
                "type": "object",
                "properties": parameterSpace,
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
        self.subGroups = subGroups

    def execute(self, params):
        # Executes the function
        if not self.computeLoss:
            testGlobals = {}
            exec(self.computeScript, testGlobals)
            self.computeLoss = testGlobals['computeLoss']
        return self.computeLoss(params)

    def convertResultsToTrials(self, results):
        trials = hyperopt.Trials()

        groupNames = ['group' + str(n+1) for n in range(len(self.subGroups))]

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

            for parameter in self.parameters:
                name = parameter['name']
                value = None
                if parameter['group'] == 'primary':
                    value = result[name]
                elif name in result['group']:
                    value = result['group'][name]
                else:
                    value = None

                if value is None:
                    data['misc']['idxs'][name] = []
                    data['misc']['vals'][name] = []
                elif name == 'activation' and isinstance(result[name], str):  # Hack here just to get it working quickly
                    values = ['relu', 'elu', "selu", "rrelu"]
                    data['misc']['idxs'][name] = [resultIndex]
                    data['misc']['vals'][name] = [values.index(value)]
                else:
                    data['misc']['idxs'][name] = [resultIndex]
                    data['misc']['vals'][name] = [value]

            data['misc']['idxs']['groupIndex'] = [resultIndex]
            data['misc']['vals']['groupIndex'] = [groupNames.index(result['group']['group'])]

            trials.insert_trial_doc(data)
        return trials

    def getATPEHyperoptSpace(self):
        return {
            "gamma": hyperopt.hp.quniform("gamma", 0.1, 2.0, 0.2),
            "nEICandidates": hyperopt.hp.qloguniform("nEICandidates", math.log(2), math.log(48), 2),
            "secondaryCutoff": hyperopt.hp.quniform("secondaryCutoff", -1, 1, 0.1),
            "secondaryCorrelationExponent": hyperopt.hp.quniform("secondaryCorrelationExponent", 1, 3, 0.5),
            "secondaryProbabilityMode": hyperopt.hp.choice("secondaryProbabilityMode", [
                {
                    "mode": "fixed",
                    "probability": hyperopt.hp.quniform("secondaryFixedProbability", 0.2, 0.8, 0.15)
                },
                {
                    "mode": "correlation",
                    "multiplier": hyperopt.hp.quniform("secondaryCorrelationMultiplier", 0.2, 1.8, 0.2)
                }
            ]),
            "secondaryLockingMode": hyperopt.hp.choice("secondaryLockingMode", [
                {
                    "locking": "top",
                    "percentile": hyperopt.hp.quniform("secondaryTopLockingPercentile", 0, 10, 5)
                },
                {
                    "locking": "random"
                }
            ]),
            "resultFilteringMode": hyperopt.hp.choice("resultFilteringMode", [
                {
                    "filtering": "none"
                },
                {
                    "filtering": "random",
                    "probability": hyperopt.hp.quniform("resultFilteringRandomProbability", 0.7, 0.9, 0.1)
                },
                {
                    "filtering": "age",
                    "multiplier": hyperopt.hp.quniform("resultFilteringAgeMultiplier", 1.0, 4.0, 1.0)
                },
                {
                    "filtering": "loss_rank",
                    "multiplier": hyperopt.hp.quniform("resultFilteringLossRankMultiplier", 1.0, 4.0, 1.0)
                }
            ])
        }

    def getFlatATPEParameters(self, params):
        return {
            "gamma": params['gamma'],
            "nEICandidates": params['nEICandidates'],
            "secondaryCutoff": params['secondaryCutoff'],
            "secondaryCorrelationExponent": params['secondaryCorrelationExponent'],
            "secondaryProbabilityMode": params['secondaryProbabilityMode']['mode'],
            "secondaryFixedProbability": params['secondaryProbabilityMode']['probability'] if params['secondaryProbabilityMode']['mode'] == 'fixed' else None,
            "secondaryCorrelationMultiplier": params['secondaryProbabilityMode']['multiplier'] if params['secondaryProbabilityMode']['mode'] == 'correlation' else None,
            "secondaryLockingMode": params['secondaryLockingMode']['locking'],
            "secondaryTopLockingPercentile": params['secondaryLockingMode']['percentile'] if params['secondaryLockingMode']['locking'] == 'top' else None,
            "resultFilteringMode": params['resultFilteringMode']['filtering'],
            "resultFilteringRandomProbability": params['resultFilteringMode']['probability'] if params['resultFilteringMode']['filtering'] == 'random' else None,
            "resultFilteringAgeMultiplier": params['resultFilteringMode']['multiplier'] if params['resultFilteringMode']['filtering'] == 'age' else None,
            "resultFilteringLossRankMultiplier": params['resultFilteringMode']['multiplier'] if params['resultFilteringMode']['filtering'] == 'loss_rank' else None
        }

    def runSearch(self, currentResults, trials=10, atpeParams=None):
        losses = []

        if atpeParams is None:
            atpeParams = {
                'gamma': 1.0,
                'nEICandidates': 24,
                'secondaryCutoff': 0,
                "secondaryCorrelationExponent": 1.0,
                "secondaryProbabilityMode": {
                    "mode": "fixed",
                    "probability": 0.0
                },
                "secondaryLockingMode": {
                    "locking": "random",
                    "percentile": 0
                },
                "resultFilteringMode": {
                    "filtering": "none"
                }
            }

        def getValue(result, parameter):
            if parameter['group'] == 'primary':
                return result[parameter['name']]
            elif parameter['name'] in result['group']:
                return result['group'][parameter['name']]
            else:
                return None

        initializationRounds = 10

        best = None
        bestLoss = None

        newResults = []

        def computePrimarySecondary():
            if len(currentResults) < initializationRounds:
                return self.parameters, [], [0.5] * len(self.parameters) # Put all parameters as primary

            if len(set(result['loss'] for result in currentResults)) < 5:
                return self.parameters, [], [0.5] * len(self.parameters)  # Put all parameters as primary

            cutoffForTrial = atpeParams['secondaryCutoff']

            def getValue(result, parameter):
                if parameter['group'] == 'primary':
                    return result[parameter['name']]
                elif parameter['name'] in result['group']:
                    return result['group'][parameter['name']]
                else:
                    return None

            totalWeight = 0
            correlations = {}
            for parameter in self.parameters:
                if len(set(getValue(result, parameter) for result in currentResults if getValue(result, parameter) is not None)) < 2:
                    correlations[parameter['name']] = 0
                else:
                    values = []
                    valueLosses = []
                    for result in currentResults:
                        if isinstance(getValue(result, parameter), float) or isinstance(getValue(result, parameter), int):
                            values.append(getValue(result, parameter))
                            valueLosses.append(result['loss'])

                    correlation = math.pow(abs(scipy.stats.spearmanr(values, valueLosses)[0]), atpeParams['secondaryCorrelationExponent'])
                    correlations[parameter['name']] = correlation
                    totalWeight += correlation

            threshold = totalWeight * abs(cutoffForTrial)

            if cutoffForTrial < 0:
                # Reverse order - we lock in the highest correlated parameters
                sortedParameters = sorted(self.parameters, key=lambda parameter: correlations[parameter['name']])
            else:
                # Normal order - sort properties by their correlation to lock in lowest correlated parameters
                sortedParameters = sorted(self.parameters, key=lambda parameter: -correlations[parameter['name']])

            primaryParameters = []
            secondaryParameters = []
            cumulative = totalWeight
            for parameter in sortedParameters:
                if cumulative < threshold:
                    secondaryParameters.append(parameter)
                else:
                    primaryParameters.append(parameter)

                cumulative -= correlations[parameter['name']]

            return primaryParameters, secondaryParameters, correlations

        while len(newResults) < trials:
            nextParams = {}

            def sample(parameters):
                nonlocal nextParams
                nextParams = parameters
                return {"loss": 0.5, 'status': 'ok'}

            lockedValues = {}
            filteredResults = []
            removedResults = []
            if len(currentResults) > initializationRounds:
                primaryParameters, secondaryParameters, correlations = computePrimarySecondary()

                sortedResults = list(sorted(list(currentResults), key=lambda result:(result['loss'] if result['loss'] is not None else 1.0)))
                topResults = sortedResults
                if atpeParams['secondaryLockingMode']['locking'] == 'top':
                    topResultsN = max(1, int(math.ceil(len(sortedResults) * atpeParams['secondaryLockingMode']['percentile'] / 100.0)))
                    topResults = sortedResults[:topResultsN]

                if best is not None:
                    # Any secondary parameters have may be locked to either the current best value or any value within the result pool.
                    for secondary in secondaryParameters:
                        if atpeParams['secondaryProbabilityMode']['mode'] == 'fixed':
                            if random.uniform(0, 1) < atpeParams['secondaryProbabilityMode']['probability']:
                                if atpeParams['secondaryLockingMode']['locking'] == 'top':
                                    lockResult = random.choice(topResults)
                                    if getValue(lockResult, secondary) is not None:
                                        lockedValues[secondary['name']] = getValue(lockResult, secondary)
                                elif atpeParams['secondaryLockingMode']['locking'] == 'random':
                                    if 'rounding' in secondary['space']:
                                        lockedValues[secondary['name']] = random.choice(numpy.linspace(0.0, 1.0, num=(secondary['cardinality']+1)))
                                    else:
                                        lockedValues[secondary['name']] = random.uniform(0.0, 1.0)
                        elif atpeParams['secondaryProbabilityMode']['mode'] == 'correlation':
                            probability = max(0, min(1, abs(correlations[secondary['name']]) * atpeParams['secondaryProbabilityMode']['multiplier']))
                            if random.uniform(0, 1) < probability:
                                if atpeParams['secondaryLockingMode']['locking'] == 'top':
                                    lockResult = random.choice(topResults)
                                    if getValue(lockResult, secondary) is not None:
                                        lockedValues[secondary['name']] = getValue(lockResult, secondary)
                                elif atpeParams['secondaryLockingMode']['locking'] == 'random':
                                    if 'rounding' in secondary['space']:
                                        lockedValues[secondary['name']] = random.choice(numpy.linspace(0.0, 1.0, num=(secondary['cardinality']+1)))
                                    else:
                                        lockedValues[secondary['name']] = random.uniform(0.0, 1.0)

                # Now last step, we filter results prior to sending them into ATPE
                for resultIndex, result in enumerate(currentResults):
                    if atpeParams['resultFilteringMode']['filtering'] == 'none':
                        filteredResults.append(result)
                    elif atpeParams['resultFilteringMode']['filtering'] == 'random':
                        if random.uniform(0, 1) < atpeParams['resultFilteringMode']['probability']:
                            filteredResults.append(result)
                        else:
                            removedResults.append(result)
                    elif atpeParams['resultFilteringMode']['filtering'] == 'age':
                        age = float(resultIndex) / float(len(currentResults))
                        if random.uniform(0, 1) < (atpeParams['resultFilteringMode']['multiplier'] * age):
                            filteredResults.append(result)
                        else:
                            removedResults.append(result)
                    elif atpeParams['resultFilteringMode']['filtering'] == 'loss_rank':
                        rank = 1.0 - (float(sortedResults.index(result)) / float(len(currentResults)))
                        if random.uniform(0, 1) < (atpeParams['resultFilteringMode']['multiplier'] * rank):
                            filteredResults.append(result)
                        else:
                            removedResults.append(result)

            # If we are in initialization, or by some other fluke of random nature that we end up with no results after filtering,
            # then just use all the results
            if len(filteredResults) == 0:
                filteredResults = currentResults

            space = {}
            for parameter in self.parameters:
                if parameter['group'] == 'primary':
                    if parameter['name'] in lockedValues:
                        space[parameter['name']] = lockedValues[parameter['name']]
                    elif 'rounding' in parameter['space']:
                        space[parameter['name']] = hyperopt.hp.quniform(parameter['name'], 0, 1, parameter['space']['rounding'])
                    else:
                        space[parameter['name']] = hyperopt.hp.uniform(parameter['name'], 0, 1)

            groups = []
            for groupIndex, group in enumerate(self.subGroups):
                groupSpace = {"group": 'group' + str(groupIndex+1)}
                for parameter in group['parameters']:
                    if parameter['name'] in lockedValues:
                        groupSpace[parameter['name']] = lockedValues[parameter['name']]
                    elif 'rounding' in parameter['space']:
                        groupSpace[parameter['name']] = hyperopt.hp.quniform(parameter['name'], 0, 1, parameter['space']['rounding'])
                    else:
                        groupSpace[parameter['name']] = hyperopt.hp.uniform(parameter['name'], 0, 1)
                groups.append(groupSpace)

            space['group'] = hyperopt.hp.choice('groupIndex', groups)

            hyperopt.fmin(fn=sample,
                          space=space,
                          algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=initializationRounds, gamma=atpeParams['gamma'], n_EI_candidates=int(atpeParams['nEICandidates'])),
                          max_evals=1,
                          trials=self.convertResultsToTrials(filteredResults),
                          rstate=numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1))))

            result = self.execute(nextParams)
            loss = result['loss']
            losses.append(loss)
            data = dict(nextParams)  # Clone the next dict
            data['loss'] = loss
            data['trial'] = len(currentResults)
            currentResults.append(data)
            newResults.append(data)

            if best is None or loss < bestLoss:
                best = data
                bestLoss = loss
        return best, newResults

    def computeCardinality(self):
        if self.log10_cardinality: # Return cached computation
            return self.log10_cardinality

        log10_cardinality = 0
        for param in self.parameters:
            if param['group'] == 'primary':
                log10_cardinality += math.log10(float(param['cardinality']))

        group_log10_cardinality = None
        for group in self.subGroups:
            subgroup_log10_cardinality = 0
            for param in group['parameters']:
                subgroup_log10_cardinality += math.log10(float(param['cardinality']))

            # Combine the log cardinalities in a numerically stable way, taking advantage of logarithm identities.
            #  Check here to obesrve: https://www.desmos.com/calculator/efkbbftd18 to observe
            if group_log10_cardinality is None:
                group_log10_cardinality = subgroup_log10_cardinality
            elif (group_log10_cardinality - subgroup_log10_cardinality) > 3:
                group_log10_cardinality = group_log10_cardinality + 1
            elif (group_log10_cardinality - subgroup_log10_cardinality) < 3:
                group_log10_cardinality = subgroup_log10_cardinality + 1
            else:
                group_log10_cardinality = subgroup_log10_cardinality + math.log10(1 + math.pow(10, group_log10_cardinality - subgroup_log10_cardinality))

        log10_cardinality += group_log10_cardinality

        self.log10_cardinality = log10_cardinality

        return self.log10_cardinality

    def computeBestATPEParamsAtHistory(self, history, historyIndex, atpeSearchLength, length):
        start = datetime.datetime.now()

        # Compute stats
        stats = self.computeAllResultStatistics(history)

        trialATPEParamResults = []

        # Create a function which evaluates ATPE parameters, given a history
        def evaluateATPEParameters(history, length, atpeParameters):
            # pprint(atpeParameters)
            # Copy the history, so it doesn't get modified
            best, _ = self.runSearch(currentResults=copy.deepcopy(history), trials=length, atpeParams=atpeParameters)
            data = {
                'loss': best['loss'],
                'atpeParams': atpeParameters
            }
            trialATPEParamResults.append(data)
            return best['loss']

        dict(hyperopt.fmin(fn=functools.partial(evaluateATPEParameters, history, length),
                           space=self.getATPEHyperoptSpace(),
                           algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=10, gamma=1.0, n_EI_candidates=4),
                           max_evals=atpeSearchLength,
                           rstate=numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1)))))

        best = min(trialATPEParamResults, key=lambda result: result['loss'])
        flatAtpeParams = self.getFlatATPEParameters(best['atpeParams'])
        for param in flatAtpeParams.keys():
            stats[param] = flatAtpeParams[param]

        stats['trial'] = len(history)
        stats['log10_trial'] = math.log10(len(history))
        stats['history'] = historyIndex
        stats['loss'] = best['loss']
        stats['time'] = (datetime.datetime.now() - start).total_seconds()
        return stats, trialATPEParamResults

    def computeOptimizationResults(self, number_histories=10, trial_lengths=None, atpeSearchLength = 1000, verbose=False, processExecutor=None):
        if trial_lengths is None:
            trial_lengths = [10,10,10,10,10,10,10,10,10,25,25,25,25,25,25]

        # Construct 10 different result histories. Each history will represent a different convergence speed, with different assumptions on how good our ATPE
        # parameters were at that point in the history
        histories = [[] for history in range(number_histories)]

        # Seed each history with 10 random results, representing the initialization period
        for history in histories:
            best, newResults = self.runSearch(currentResults=[], trials=10, atpeParams=None)
            for result in newResults:
                history.append(result)

        optimizationResults = []

        # Loop for each of our trial lengths
        for length in trial_lengths:
            # Now for each history, find the optimal ATPE parameters.
            allATPEParamResultFutures = []
            for historyIndex, history in enumerate(histories):
                self.computeLoss = None # Delete the computeLoss function - it can't be pickled. But it can just be recreated from the computeScript
                allATPEParamResultFutures.append(processExecutor.submit(self.computeBestATPEParamsAtHistory, history, historyIndex, atpeSearchLength, length))

            allATPEParamResults = []
            for future in concurrent.futures.as_completed(allATPEParamResultFutures):
                stats, trialATPEParamResults = future.result()
                allATPEParamResults = allATPEParamResults + trialATPEParamResults
                optimizationResults.append(stats)
                if verbose:
                    pprint(stats)
                    sys.stdout.flush()
                    sys.stderr.flush()

            # Now we extend each history by using ATPE parameters at various percentiles
            sortedATPEParamResults = sorted(allATPEParamResults, key=lambda result: result['loss'])

            for historyIndex, history in enumerate(histories):
                percentile = (float(historyIndex+1) / float(len(histories))) * 0.5
                index = int(percentile * float(len(histories)))
                atpeParamsForExtension = sortedATPEParamResults[index]['atpeParams']
                # print(atpeParamsForExtension)
                best, newResults = self.runSearch(currentResults=copy.deepcopy(history), trials=length, atpeParams=atpeParamsForExtension)
                for result in newResults:
                    history.append(result)
        return optimizationResults


    def addAlgorithmStatistics(self, data):
        data['num_parameters'] = len(self.parameters)
        data['log10_cardinality'] = self.computeCardinality()
        data['noise'] = self.noiseFactor
        data['fail_rate'] = self.failRate

        data['interactions'] = float(self.interactionCounts['linear'] + self.interactionCounts['peakvalley'] + self.interactionCounts['wave'] + self.interactionCounts['random'])
        data['interactions_linear'] = float(self.interactionCounts['linear']) / max(1, data['interactions'])
        data['interactions_peakvalley'] = float(self.interactionCounts['peakvalley']) / max(1, data['interactions'])
        data['interactions_wave'] = float(self.interactionCounts['wave']) / max(1, data['interactions'])
        data['interactions_random'] = float(self.interactionCounts['random']) / max(1, data['interactions'])
        data['interactions_index'] = float(max(1, data['interactions'])) / float(data['num_parameters'] * data['num_parameters'] * 0.3)

        data['contributions_linear'] = float(self.contributionCounts['linear']) / data['num_parameters']
        data['contributions_peakvalley'] = float(self.contributionCounts['peakvalley']) / data['num_parameters']
        data['contributions_exponential'] = float(self.contributionCounts['exponential']) / data['num_parameters']
        data['contributions_logarithmic'] = float(self.contributionCounts['logarithmic']) / data['num_parameters']
        data['contributions_random'] = float(self.contributionCounts['random']) / data['num_parameters']

    def computePartialResultStatistics(self, results):
        losses = numpy.array(sorted([result['loss'] for result in results if result['loss'] is not None]))

        bestLoss = 0
        percentile5Loss = 0
        percentile25Loss = 0
        percentile50Loss = 0
        percentile75Loss = 0
        statistics = {}

        if len(set(losses)) > 1:
            bestLoss = numpy.percentile(losses, 0)
            percentile5Loss = numpy.percentile(losses, 5)
            percentile25Loss = numpy.percentile(losses, 25)
            percentile50Loss = numpy.percentile(losses, 50)
            percentile75Loss = numpy.percentile(losses, 75)

            statistics['loss_skew'] = scipy.stats.skew(losses)
            statistics['loss_kurtosis'] = scipy.stats.kurtosis(losses)
        else:
            statistics['loss_skew'] = 0
            statistics['loss_kurtosis'] = 0

        if percentile50Loss == 0:
            statistics['loss_stddev_median_ratio'] = 0
            statistics['loss_best_percentile50_ratio'] = 0
        else:
            statistics['loss_stddev_median_ratio'] = numpy.std(losses) / percentile50Loss
            statistics['loss_best_percentile50_ratio'] = bestLoss / percentile50Loss

        if bestLoss == 0:
            statistics['loss_stddev_best_ratio'] = 0
        else:
            statistics['loss_stddev_best_ratio'] = numpy.std(losses) / bestLoss

        if percentile25Loss == 0:
            statistics['loss_best_percentile25_ratio'] = 0
            statistics['loss_percentile5_percentile25_ratio'] = 0
        else:
            statistics['loss_best_percentile25_ratio'] = bestLoss / percentile25Loss
            statistics['loss_percentile5_percentile25_ratio'] = percentile5Loss / percentile25Loss

        if percentile75Loss == 0:
            statistics['loss_best_percentile75_ratio'] = 0
        else:
            statistics['loss_best_percentile75_ratio'] = bestLoss / percentile75Loss

        def getValue(result, parameter):
            if parameter['group'] == 'primary':
                return result[parameter['name']]
            elif parameter['name'] in result['group']:
                return result['group'][parameter['name']]
            else:
                return None

        # Now we compute correlations between each parameter and the loss
        correlations = []
        for parameter in self.parameters:
            if len(set(getValue(result, parameter) for result in results if getValue(result, parameter) is not None)) < 2:
                correlations.append(0)
            else:
                values = []
                valueLosses = []
                for result in results:
                    if isinstance(getValue(result, parameter), float) or isinstance(getValue(result, parameter), int):
                        values.append(getValue(result, parameter))
                        valueLosses.append(result['loss'])

                correlation = abs(scipy.stats.spearmanr(values, valueLosses)[0])
                correlations.append(correlation)

        correlations = numpy.array(correlations)

        if len(set(correlations)) == 1:
            statistics['correlation_skew'] = 0
            statistics['correlation_kurtosis'] = 0
            statistics['correlation_stddev_median_ratio'] = 0
            statistics['correlation_stddev_best_ratio'] = 0

            statistics['correlation_best_percentile25_ratio'] = 0
            statistics['correlation_best_percentile50_ratio'] = 0
            statistics['correlation_best_percentile75_ratio'] = 0
            statistics['correlation_percentile5_percentile25_ratio'] = 0
        else:
            bestCorrelation = numpy.percentile(correlations, 100) # Correlations are in the opposite order of losses, higher correlation is considered "best"
            percentile5Correlation = numpy.percentile(correlations, 95)
            percentile25Correlation = numpy.percentile(correlations, 75)
            percentile50Correlation = numpy.percentile(correlations, 50)
            percentile75Correlation = numpy.percentile(correlations, 25)

            statistics['correlation_skew'] = scipy.stats.skew(correlations)
            statistics['correlation_kurtosis'] = scipy.stats.kurtosis(correlations)

            if percentile50Correlation == 0:
                statistics['correlation_stddev_median_ratio'] = 0
                statistics['correlation_best_percentile50_ratio'] = 0
            else:
                statistics['correlation_stddev_median_ratio'] = numpy.std(correlations) / percentile50Correlation
                statistics['correlation_best_percentile50_ratio'] = bestCorrelation / percentile50Correlation

            if bestCorrelation == 0:
                statistics['correlation_stddev_best_ratio'] = 0
            else:
                statistics['correlation_stddev_best_ratio'] = numpy.std(correlations) / bestCorrelation

            if percentile25Correlation == 0:
                statistics['correlation_best_percentile25_ratio'] = 0
                statistics['correlation_percentile5_percentile25_ratio'] = 0
            else:
                statistics['correlation_best_percentile25_ratio'] = bestCorrelation / percentile25Correlation
                statistics['correlation_percentile5_percentile25_ratio'] = percentile5Correlation / percentile25Correlation

            if percentile75Correlation == 0:
                statistics['correlation_best_percentile75_ratio'] = 0
            else:
                statistics['correlation_best_percentile75_ratio'] = bestCorrelation / percentile75Correlation

        return statistics

    def computeAllResultStatistics(self, results):
        losses = numpy.array(sorted([result['loss'] for result in results if result['loss'] is not None]))

        if len(set(losses)) > 1:
            percentile10Loss = numpy.percentile(losses, 10)
            percentile20Loss = numpy.percentile(losses, 20)
            percentile30Loss = numpy.percentile(losses, 30)
        else:
            percentile10Loss = losses[0]
            percentile20Loss = losses[0]
            percentile30Loss = losses[0]

        allResults = list(results)
        percentile10Results = [result for result in results if result['loss'] is not None and result['loss'] <= percentile10Loss]
        percentile20Results = [result for result in results if result['loss'] is not None and result['loss'] <= percentile20Loss]
        percentile30Results = [result for result in results if result['loss'] is not None and result['loss'] <= percentile30Loss]

        recent10Count = min(len(results), 10)
        recent10Results = results[-recent10Count:]

        recent25Count = min(len(results), 25)
        recent25Results = results[-recent25Count:]

        recent15PercentCount = max(math.ceil(len(results)*0.15), 5)
        recent15PercentResults = results[-recent15PercentCount:]

        statistics = {}
        allResultStatistics = self.computePartialResultStatistics(allResults)
        for stat,value in allResultStatistics.items():
            statistics['all_' + stat] = value

        percentile10Statistics = self.computePartialResultStatistics(percentile10Results)
        for stat,value in percentile10Statistics.items():
            statistics['top_10%_' + stat] = value

        percentile20Statistics = self.computePartialResultStatistics(percentile20Results)
        for stat,value in percentile20Statistics.items():
            statistics['top_20%_' + stat] = value

        percentile30Statistics = self.computePartialResultStatistics(percentile30Results)
        for stat,value in percentile30Statistics.items():
            statistics['top_30%_' + stat] = value

        recent10Statistics = self.computePartialResultStatistics(recent10Results)
        for stat,value in recent10Statistics.items():
            statistics['recent_10_' + stat] = value

        recent25Statistics = self.computePartialResultStatistics(recent25Results)
        for stat,value in recent25Statistics.items():
            statistics['recent_25_' + stat] = value

        recent15PercentResult = self.computePartialResultStatistics(recent15PercentResults)
        for stat,value in recent15PercentResult.items():
            statistics['recent_15%_' + stat] = value

        self.addAlgorithmStatistics(statistics)

        # Although we have added lots of protection in the computePartialResultStatistics code, one last hedge against any NaN or infinity values coming up
        # in our statistics
        for key in statistics.keys():
            if math.isnan(statistics[key]) or math.isinf(statistics[key]):
                statistics[key] = 0

        return statistics


    def computeBasicStatistics(self):
        best, results = self.runSearch(currentResults=[], trials=100, atpeParams=None) # Use default atpe params
        return self.computeAllResultStatistics(results)


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


def createContributionChartExample(type=4):
    algo = AlgorithmSimulation()
    param1 = algo.createHyperParameter()
    contribution = algo.createHyperParameterContribution(param1, type=type)

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

def computeStats(algo):
    stats = algo.computeBasicStatistics()
    algo.computeLoss = None
    return (stats, algo)

def chooseAlgorithmsForTest(total, shrinkage=0.2, processExecutor=None):
    parameterSpacesToConsider = int(math.ceil(float(total) / shrinkage))
    numberFinalParameterSpaces = total

    resultFutures = []
    for n in range(parameterSpacesToConsider):
        algo = AlgorithmSimulation()
        resultFutures.append(processExecutor.submit(computeStats, algo))

    results = []
    for n, future in enumerate(concurrent.futures.as_completed(resultFutures)):
        stats, algo = future.result()
        fileName = 'algo' + str(n) + ".bin"
        stats['fileName'] = fileName
        with open(fileName, "wb") as file:
            algo.computeLoss = None
            pickle.dump({
                "algo": algo,
                "stats": stats
            }, file)

        results.append(stats)

        with open("algorithms.csv", "wt") as file:
            writer = csv.DictWriter(file, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerows(results)
        print("Completed algorithm " + str(n))
        sys.stdout.flush()
        sys.stderr.flush()

    statKeys = [stat for stat in results[0].keys() if stat != 'fileName']

    # Prepare our vectors
    vectors = []
    for result in results:
        vector = [result[stat] for stat in statKeys]
        vectors.append(vector)

    # Normalize
    vectors = sklearn.preprocessing.scale(vectors)

    # Add weights
    for index, stat in enumerate(statKeys):
        if stat == 'log10_cardinality':
            vectors[:,index] *= 25
        elif stat == 'num_parameters':
            vectors[:,index] *= 10
        elif stat == 'noise':
            vectors[:,index] *= 10
        elif stat == 'fail_rate':
            vectors[:,index] *= 5
        elif 'interaction' in stat:
            vectors[:, index] *= 2
        elif 'contribution' in stat:
            vectors[:,index] *= 2
        else:
            base = None
            if 'all_' in stat or 'top_10%' in stat:
                base = 3
            if 'recent_10' in stat or 'recent_25' in stat or 'top_20%' in stat:
                base = 2
            if 'top_30%' in stat in stat or 'recent_15%' in stat:
                base = 1
            if base is None:
                print(stat)

            if 'skew' in stat or 'kurtosis' in stat:
                vectors[:, index] *= (3 * base)
            elif 'stddev' in stat:
                vectors[:, index] *= (2 * base)
            else:
                vectors[:, index] *= (1 * base)

    print("Clustering the hyper-parameter spaces")
    # Cluster
    cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=numberFinalParameterSpaces, affinity='euclidean', linkage='complete')
    labels = cluster.fit_predict(vectors)

    grouped = {}
    for index, result in enumerate(results):
        label = labels[index]

        if label not in grouped:
            grouped[label] = [result]
        else:
            grouped[label].append(result)

    chosenSpaces = []
    for key,group in grouped.items():
        chosenSpaces.append(random.choice(group))

    with open("chosen_algorithms.csv", "wt") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(chosenSpaces)

    #
    # fig, axes = plt.subplots(2)
    #
    #
    # skews = [result['all_loss_skew'] for result in chosenSpaces]
    # kurtosis = [result['all_loss_kurtosis'] for result in chosenSpaces]
    #
    # axes[0].scatter(skews, kurtosis)
    #
    # skews = [result['all_loss_skew'] for result in results]
    # kurtosis = [result['all_loss_kurtosis'] for result in results]
    #
    # axes[1].scatter(skews, kurtosis)
    # plt.show()

    print("Finished deciding hyperparameter spaces for test")

    return chosenSpaces


def testAlgo(algo, algoInfo, processExecutor, trialLengths, verbose): # We have to put it in this form so its compatible with processExecutor
    return (algo.computeOptimizationResults(trial_lengths=trialLengths, number_histories=5, atpeSearchLength=100, verbose=verbose, processExecutor=processExecutor), algoInfo)

if __name__ == '__main__':
    verbose = True

    trialsLength = 500
    ratio = 1.15
    currentTrialLength = 5.0
    totalTrials = 5
    trialLengths = [5]
    # Construct a series of trial lengths until we get to our target, as a geometric sequence
    while totalTrials < trialsLength:
        currentTrialLength *= ratio
        totalTrials += int(currentTrialLength)
        trialLengths.append([int(currentTrialLength)])

    algorithmsAtOnce = int(math.ceil(float(default_max_workers) / 5.0))

    with concurrent.futures.ThreadPoolExecutor(max_workers=algorithmsAtOnce) as threadExecutor:
        with concurrent.futures.ProcessPoolExecutor(max_workers=default_max_workers) as processExecutor:
            resultFutures = []

            chosen = chooseAlgorithmsForTest(total=250, processExecutor=processExecutor)
            random.shuffle(chosen) # Shuffle them for extra randomness
            for index, algoInfo in enumerate(chosen):
                with open(algoInfo['fileName'], "rb") as file:
                    data = pickle.load(file)
                    algo = data['algo']

                resultFutures.append(threadExecutor.submit(testAlgo, algo, algoInfo, processExecutor,trialLengths, verbose))

            results = []
            for future in concurrent.futures.as_completed(resultFutures):
                futureResult = future.result()
                algoInfo = futureResult[1]
                algoResults = futureResult[0]

                for result in algoResults:
                    result['algorithm'] = algoInfo['fileName']
                    results.append(result)

                with open('results.csv', 'wt') as file:
                    writer = csv.DictWriter(file, fieldnames=results[0])
                    writer.writeheader()
                    writer.writerows(results)
                if verbose:
                    pprint(algoResults)
                    sys.stdout.flush()
                    sys.stderr.flush()
                else:
                    print("Completed Analysis for algorithm ", algoInfo['fileName'])
