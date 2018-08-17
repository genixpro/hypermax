from hypermax.hyperparameter import  Hyperparameter


class Configuration:
    def __init__(self, data):
        self.data = data



    def createHyperparameterSpace(self, lockedValues=None):
        param = Hyperparameter(self.data['hyperparameters'])

        space = param.createHyperoptSpace(lockedValues)

        return space

