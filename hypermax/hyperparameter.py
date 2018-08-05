from hyperopt import hp
import math


class Hyperparameter:
    """ Represents a hyperparameter being options."""

    def __init__(self, config, root='root'):
        self.config = config
        self.root = root



    def createHyperoptSpace(self):
        name = self.root

        if 'anyOf' in self.config or 'oneOf' in self.config:
            data = []
            if 'anyOf' in self.config:
                data = self.config['anyOf']
            else:
                data = self.config['oneOf']

            choices = hp.choice(name, [Hyperparameter(param, name + "." + str(index)).createHyperoptSpace() for index,param in enumerate(data)])

            return choices
        elif self.config['type'] == 'object':
            space = {}
            for key in self.config['properties'].keys():
                config = self.config['properties'][key]
                space[key] = Hyperparameter(config, name + "." + key).createHyperoptSpace()
            return space
        elif self.config['type'] == 'number':
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
                subKeys = Hyperparameter(param, name + "." + str(index)).getFlatParameterNames()
                for key in subKeys:
                    keys.add(key)

            return keys
        elif self.config['type'] == 'object':
            keys = set()
            for key in self.config['properties'].keys():
                config = self.config['properties'][key]
                subKeys = Hyperparameter(config, name + "." + key).getFlatParameterNames()
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
                subParameters = Hyperparameter(param, name + "." + str(index)).getFlatParameters()
                parameters = parameters + subParameters
            return parameters
        elif self.config['type'] == 'object':
            parameters = []
            for key in self.config['properties'].keys():
                config = self.config['properties'][key]
                subParameters = Hyperparameter(config, name + "." + key).getFlatParameters()
                parameters = parameters + subParameters
            return parameters
        elif self.config['type'] == 'number':
            return [self]
