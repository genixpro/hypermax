from hyperopt import hp
import math
from pprint import pprint
import re


class Hyperparameter:
    """ Represents a hyperparameter being options."""

    def __init__(self, config, parent=None, root='root'):
        self.config = config
        self.root = root
        self.name = root[5:]
        self.parent = parent
        self.resultVariableName = re.sub("\\.\\d+\\.", ".", self.name)

    def createHyperoptSpace(self, lockedValues=None):
        name = self.root

        if lockedValues is None:
            lockedValues = {}

        if 'anyOf' in self.config or 'oneOf' in self.config:
            data = []
            if 'anyOf' in self.config:
                data = self.config['anyOf']
            else:
                data = self.config['oneOf']

            subSpaces = [Hyperparameter(param, self, name + "." + str(index)).createHyperoptSpace(lockedValues) for index,param in enumerate(data)]
            for index, space in enumerate(subSpaces):
                space["$index"] = index

            choices = hp.choice(name, subSpaces)

            return choices
        elif 'enum' in self.config:
            if self.name in lockedValues:
                return lockedValues[self.name]

            choices = hp.choice(name, self.config['enum'])
            return choices
        elif 'constant' in self.config:
            if self.name in lockedValues:
                return lockedValues[self.name]

            return self.config['constant']
        elif self.config['type'] == 'object':
            space = {}
            for key in self.config['properties'].keys():
                config = self.config['properties'][key]
                space[key] = Hyperparameter(config, self, name + "." + key).createHyperoptSpace(lockedValues)
            return space
        elif self.config['type'] == 'number':
            if self.name in lockedValues:
                return lockedValues[self.name]

            mode = self.config.get('mode', 'uniform')
            scaling = self.config.get('scaling', 'linear')

            if mode == 'uniform':
                min = self.config.get('min', 0)
                max = self.config.get('max', 1)
                rounding = self.config.get('rounding', None)

                if scaling == 'linear':
                    if rounding is not None:
                        return hp.quniform(name, min, max, rounding)
                    else:
                        return hp.uniform(name, min, max)
                elif scaling == 'logarithmic':
                    if rounding is not None:
                        return hp.qloguniform(name, math.log(min), math.log(max), rounding)
                    else:
                        return hp.loguniform(name, math.log(min), math.log(max))
            if mode == 'normal':
                mean = self.config.get('mean', 0)
                stddev = self.config.get('stddev', 1)
                rounding = self.config.get('rounding', None)

                if scaling == 'linear':
                    if rounding is not None:
                        return hp.qnormal(name, mean, stddev, rounding)
                    else:
                        return hp.normal(name, mean, stddev)
                elif scaling == 'logarithmic':
                    if rounding is not None:
                        return hp.qlognormal(name, math.log(mean), math.log(stddev), rounding)
                    else:
                        return hp.lognormal(name, math.log(mean), math.log(stddev))

    def getFlatParameterNames(self):
        name = self.root

        if 'anyOf' in self.config or 'oneOf' in self.config:
            keys = set()
            if 'anyOf' in self.config:
                data = self.config['anyOf']
            else:
                data = self.config['oneOf']
                
            for index, param in enumerate(data):
                subKeys = Hyperparameter(param, self, name + "." + str(index)).getFlatParameterNames()
                for key in subKeys:
                    keys.add(key)

            return keys
        elif 'enum' in self.config or 'constant' in self.config:
            return [name]
        elif self.config['type'] == 'object':
            keys = set()
            for key in self.config['properties'].keys():
                config = self.config['properties'][key]
                subKeys = Hyperparameter(config, self, name + "." + key).getFlatParameterNames()
                for key in subKeys:
                    keys.add(key)

            return keys
        elif self.config['type'] == 'number':
            return [name]

    def getFlatParameters(self):
        name = self.root
        if 'anyOf' in self.config or 'oneOf' in self.config:
            parameters = []
            if 'anyOf' in self.config:
                data = self.config['anyOf']
            else:
                data = self.config['oneOf']

            for index, param in enumerate(data):
                subParameters = Hyperparameter(param, self, name + "." + str(index)).getFlatParameters()
                parameters = parameters + subParameters
            return parameters
        elif 'enum' in self.config or 'constant' in self.config:
            return [self]
        elif self.config['type'] == 'object':
            parameters = []
            for key in self.config['properties'].keys():
                config = self.config['properties'][key]
                subParameters = Hyperparameter(config, self, name + "." + key).getFlatParameters()
                parameters = parameters + subParameters
            return parameters
        elif self.config['type'] == 'number':
            return [self]


    def getLog10Cardinality(self):
        if 'anyOf' in self.config or 'oneOf' in self.config:
            if 'anyOf' in self.config:
                data = self.config['anyOf']
            else:
                data = self.config['oneOf']

            log10_cardinality = Hyperparameter(data[0], self, self.root + ".0").getLog10Cardinality()
            for index,subParam in enumerate(data[1]):
                # We used logarithm identities to create this reduction formula
                other_log10_cardinality = Hyperparameter(subParam, self, self.root + "." + str(index)).getLog10Cardinality()

                # Revert to linear at high and low values, for numerical stability. Check here: https://www.desmos.com/calculator/efkbbftd18 to observe
                if (log10_cardinality-other_log10_cardinality) > 3:
                    return log10_cardinality+1
                elif (log10_cardinality-other_log10_cardinality) < 3:
                    return other_log10_cardinality+1
                else:
                    return other_log10_cardinality + math.log10(1 + math.pow(10, log10_cardinality-other_log10_cardinality))
        elif 'enum' in self.config:
            return math.log10(len(self.config['enum']))
        elif 'constant' in self.config:
            return math.log10(1)
        elif self.config['type'] == 'object':
            log10_cardinality = 0
            for index,subParam in enumerate(self.config['properties'].values()):
                subParameter = Hyperparameter(subParam, self, self.root + "." + str(index))
                log10_cardinality += subParameter.getLog10Cardinality()
            return log10_cardinality
        elif self.config['type'] == 'number':
            if 'rounding' in self.config:
                return math.log10(min(20, (self.config['max'] - self.config['min']) / self.config['rounding'] + 1))
            else:
                return math.log10(20) # Default of 20 for fully uniform numbers.

    def convertToTrialValues(self, result):
        flatValues = {}

        if 'anyOf' in self.config or 'oneOf' in self.config:
            if 'anyOf' in self.config:
                data = self.config['anyOf']
            else:
                data = self.config['oneOf']

            subParameterIndex = result[self.resultVariableName + '.$index']
            flatValues[self.name] = subParameterIndex

            for index, param in enumerate(data):
                subParameter = Hyperparameter(param, self, self.root + "." + str(index))

                if index == subParameterIndex:
                    subFlatValues = subParameter.convertToTrialValues(result)
                    for key in subFlatValues:
                        flatValues[key] = subFlatValues[key]
                else:
                    for flatParam in subParameter.getFlatParameters():
                        flatValues[flatParam.name] = ""

            return flatValues
        elif 'constant' in self.config:
            flatValues[self.name] = result[self.resultVariableName]
            return flatValues
        elif 'enum' in self.config:
            flatValues[self.name] = self.config['enum'].index(result[self.resultVariableName])
            return flatValues
        elif self.config['type'] == 'object':
            for key in self.config['properties'].keys():
                config = self.config['properties'][key]

                subFlatValues = Hyperparameter(config, self, self.root + "." + key).convertToTrialValues(result)

                for key in subFlatValues:
                    flatValues[key] = subFlatValues[key]

            return flatValues
        elif self.config['type'] == 'number':
            flatValues[self.name] = result[self.resultVariableName]
            return flatValues

    def getValueFromFlatResult(self, result):
        name = self.root

        value = None

        if 'anyOf' in self.config or 'oneOf' in self.config:
            return result[self.root]['$index']
        elif 'constant' in self.config:
            return result[self.root]
        elif self.config['type'] == 'object':
            return result[self.config.root]
        elif self.config['type'] == 'number':
            return result[self.config.root]
