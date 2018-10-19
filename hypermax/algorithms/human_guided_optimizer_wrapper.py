from .optimization_algorithm_base import OptimizationAlgorithmBase
import hyperopt
import functools
import random
import numpy
import math
import numpy.random
from pprint import pprint
from hypermax.hyperparameter import Hyperparameter

class HumanGuidedOptimizerWrapper(OptimizationAlgorithmBase):
    """ This class wraps an optimizer to allow a human to provide additional guidance to it."""

    def __init__(self, baseOptimizer):
        super()
        self.baseOptimizer = baseOptimizer
        self.guidanceOptions = {
            'filteringMode': 'none',
            'filteringPercentile': 0,
            'lockedParameters': [],
            'refitParameters': [],
            'scrambleParameters': []
        }

    @classmethod
    def configurationSchema(self):
        """ This method returns the configuration schema for the human guidance options."""
        return {
            "type": "object",
            "properties": {
                "filteringMode": {
                    "type": "string",
                    "enum": ['none', 'age', 'lossrank']
                },
                "filteringPercentile": {
                    "type": "number",
                    "min": 0,
                    "max": 100
                },
                "lockedParameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "parameters": {
                            "variable": {
                                "type": "string"
                            },
                            "value": {
                            }
                        }
                    }
                },
                "refitParameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "parameters": {
                            "variable": {
                                "type": "string"
                            },
                            "refitStartTrial": {
                                "type": "number"
                            }
                        }
                    }
                },
                "scrambleParameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "parameters": {
                            "variable": {
                                "type": "string"
                            }
                        }
                    }
                },
            }
        }


    def filterHyperparameterSpace(self, hyperparameterSpace, filterEntries, root=""):
        if 'anyOf' in hyperparameterSpace or 'oneOf' in hyperparameterSpace:
            if 'anyOf' in hyperparameterSpace:
                data = hyperparameterSpace['anyOf']
            else:
                data = hyperparameterSpace['oneOf']

            newParams = []
            for index, param in enumerate(data):
                newParam = self.filterHyperparameterSpace(param, filterEntries, (root + "." if root else "") + str(index))
                newParams.append(newParam)
            if 'anyOf' in hyperparameterSpace:
                return {"anyOf": newParams}
            else:
                return {"oneOf": newParams}
        elif 'enum' in hyperparameterSpace:
            return hyperparameterSpace
        elif hyperparameterSpace['type'] == 'object':
            newProperties = {}
            for key in hyperparameterSpace['properties'].keys():
                name = root + "." + key
                if name not in filterEntries:
                    config = hyperparameterSpace['properties'][key]
                    newProperty = self.filterHyperparameterSpace(config, filterEntries, (root + "." if root else "") + key)
                    newProperties[key] = newProperty
            return {"type": "object", "properties": newProperties}
        else:
            return hyperparameterSpace

    def filterResults(self, results, filterEntries):
        newResults = []
        for result in results:
            filteredResult = {}
            for key in result:
                if (key not in filterEntries) or (key in self.resultInformationKeys):
                    filteredResult[key] = result[key]
            newResults.append(filteredResult)
        return newResults

    def createFlatParameterValues(self, parameters, hyperparameterSpace, root=""):
        if 'anyOf' in hyperparameterSpace or 'oneOf' in hyperparameterSpace:
            return {root: hyperparameterSpace}
        elif 'enum' in hyperparameterSpace:
            return {root: hyperparameterSpace}
        elif hyperparameterSpace['type'] == 'object':
            flatParams = {}
            for key in hyperparameterSpace['properties'].keys():
                # print("key", key)
                config = hyperparameterSpace['properties'][key]
                subFlatParams = self.createFlatParameterValues(parameters[key], config, root + "." + key)
                # print("subFlatParams", subFlatParams)
                for newKey in subFlatParams:
                    flatParams[newKey] = subFlatParams[newKey]

            return flatParams
        else:
            return {root: parameters}

    def recommendNextParameters(self, hyperparameterSpace, results, lockedValues=None):
        if lockedValues is None:
            lockedValues = {}

        for lockedParam in self.guidanceOptions['lockedParameters']:
            lockedValues[lockedParam['variable']] = lockedParam['value']

        for scrambledParam in self.guidanceOptions['scrambleParameters']:
            parameters = Hyperparameter(hyperparameterSpace).getFlatParameters()

            parameter = [param for param in parameters if param.name == scrambledParam['variable']][0]

            minVal = parameter.config['min']
            maxVal = parameter.config['max']

            if parameter.config.get('scaling', 'linear') == 'logarithmic':
                minVal = math.log(minVal)
                maxVal = math.log(maxVal)

            value = random.uniform(minVal, maxVal)

            if parameter.config.get('scaling', 'linear') == 'logarithmic':
                value = math.exp(value)

            if 'rounding' in parameter.config:
                value = round(value / parameter.config['rounding']) * parameter.config['rounding']

            lockedValues[parameter.name] = value

        refitParameters = sorted(self.guidanceOptions['refitParameters'], key=lambda p: p['refitStartTrial'])

        refitNames = [refitParam['variable'] for refitParam in refitParameters]

        primarySpace = self.filterHyperparameterSpace(hyperparameterSpace, refitNames)

        # Filter results to remove the non primary variables
        primaryResults = self.filterResults(results, refitNames)
        recommendedParams = self.baseOptimizer.recommendNextParameters(primarySpace, primaryResults, lockedValues)

        # print(recommendedParams)

        for index, refitParam in enumerate(refitParameters):
            startTrial = refitParam['refitStartTrial']

            # For each refit parameter, we predict after locking in the all previous parameters
            remainingRefits = refitParameters[index+1:]
            # pprint("filteredSpace", filteredSpace)

            newLockedValues = self.createFlatParameterValues(recommendedParams, self.filterHyperparameterSpace(hyperparameterSpace, refitNames[index:]))

            # pprint("newLockedValues", newLockedValues)
            filteredSpace = self.filterHyperparameterSpace(hyperparameterSpace, remainingRefits)
            filteredResults = self.filterResults(results[startTrial+1:], remainingRefits)

            refitReccomendation = self.baseOptimizer.recommendNextParameters(filteredSpace, filteredResults, newLockedValues)

            recommendedParams = refitReccomendation

        return recommendedParams

