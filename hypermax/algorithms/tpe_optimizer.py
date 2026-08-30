from .optimization_algorithm_base import OptimizationAlgorithmBase
import hyperopt
import functools
import random
import numpy
import numpy.random
from hypermax.hyperparameter import Hyperparameter

class TPEOptimizer(OptimizationAlgorithmBase):

    def __init__(self, seed=None):
        self._rng = numpy.random.default_rng(seed) if seed is not None else None

    def _next_seed(self):
        if self._rng is None:
            return random.randint(1, 2 ** 32 - 1)
        return int(self._rng.integers(1, 2 ** 32))

    def recommendNextParameters(self, hyperparameterSpace, results, currentTrials, lockedValues=None):
        if lockedValues is None:
            lockedValues = {}

        rstate = numpy.random.default_rng(seed=self._next_seed())

        trials = self.convertResultsToTrials(hyperparameterSpace, results)

        space = Hyperparameter(hyperparameterSpace).createHyperoptSpace(lockedValues)

        params = {}
        def sample(parameters):
            nonlocal params
            params = parameters
            return {"loss": 0.5, 'status': 'ok'}

        hyperopt.fmin(fn=sample,
                      space=space,
                      algo=functools.partial(hyperopt.tpe.suggest, n_EI_candidates=24, gamma=0.25),
                      max_evals=1,
                      trials=trials,
                      rstate=rstate,
                      show_progressbar=False)
        return params
