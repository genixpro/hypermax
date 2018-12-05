from hypermax.optimizer import ATPEOptimizer
from hypermax.optimizer import TPEOptimizer
from hypermax.optimizer import RandomSearchOptimizer
import hpolib.benchmarks.synthetic_functions as hpobench
from hpolib.benchmarks.ml import svm_benchmark, logistic_regression
import numpy as np
from pprint import pprint

atpeOptimizer = ATPEOptimizer()
tpeOptimizer = TPEOptimizer()
randomOptimizer = RandomSearchOptimizer()

algorithms = {
    "ATPE": atpeOptimizer,
    "TPE": tpeOptimizer,
    "Random": randomOptimizer
}

# Run Scipy.minimize on artificial testfunctions

h3 = hpobench.Hartmann3()
h6 = hpobench.Hartmann6()
b = hpobench.Branin()
bo = hpobench.Bohachevsky()
cb = hpobench.Camelback()
fo = hpobench.Forrester()
gp = hpobench.GoldsteinPrice()
le = hpobench.Levy()
rb = hpobench.Rosenbrock()

logreg = svm_benchmark.SvmOnMnist()

for f in [logreg]:
    info = f.get_meta_information()

    print("=" * 50)
    print(info['name'])

    space = {
        "type": "object",
        "properties": {}
    }

    for boundIndex, bound in enumerate(info['bounds']):
        space['properties'][str(boundIndex)] = {
            "type": "number",
            "scaling": "linear",
            "mode": "uniform",
            "min": bound[0],
            "max": bound[1]
        }

    increment = 0
    for name, optimizer in algorithms.items():
        print("Optimizer", name)
        losses = []
        for round in range(1):
            best = None
            history = []
            for trial in range(100):
                params = optimizer.recommendNextParameters(space, history)
                evalParams = [params[str(boundIndex)] for boundIndex in range(len(space['properties']))]
                val = f(evalParams)
                val += increment
                print(val)
                params['loss'] = val
                params['status'] = 'ok'
                history.append(params)
                if best is None or val < best['loss']:
                    best = params
            print(round, best['loss'])
            losses.append(best['loss'])
        averageLoss = np.mean(losses)
        averageLoss -= increment
        print("Average loss: ", averageLoss)
