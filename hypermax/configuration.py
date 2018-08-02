from hypermax.hyperparameter import  Hyperparameter
from hypermax.executor import  Executor


class Configuration:
    def __init__(self, data):
        self.data = data



    def createHyperparameterSpace(self):
        param = Hyperparameter(self.data['hyperparameters'])

        space = param.createHyperoptSpace()

        return space



    def createExecutor(self):
        executor = Executor(self.data['function'])

        return executor





