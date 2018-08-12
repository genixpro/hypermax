import hyperopt
import math
import json
import random
import numpy
from scipy.stats import norm
import scipy.interpolate
from hypermax.utils import roundPrecision


class AlgorithmSimulation:
    """ This class represents a simulation of hypothetical machine learning algorithm hyper-parameter spaces.

        It is mostly used for conducting abstract research into hyper-parameter optimization.
    """

    def __init__(self):
        self.parameters = 0

    def createHyperParameter(self):
        name = 'parameter_' + str(self.parameters)
        self.parameters += 1
        return {
            "name": name,
            "weight": roundPrecision(random.uniform(0, 1)),
            "space": {
                "type": "number",
                "min": 0,
                "max": 1,
                "scaling": "linear",
                "mode": "uniform"
            }
        }

    def createHyperParameterInteraction(self, param1, param2, type=None):
        if type is None:
            type = random.randint(0, 3)
        if type == 0:
            coords = [roundPrecision(random.uniform(0, 1)) for n in range(4)]

            coords[random.randint(0,1)] = 0 # At least one of the four corners always touches 0, and one always touches 1
            coords[random.randint(2,3)] = 1.0 # At least one of the four corners always touches 0, and one always touches 1

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
                "weight": roundPrecision(random.uniform(0, 3))
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
                    "weight": roundPrecision(random.uniform(0, 3))
                }
            else:
                return {
                    "type": "hole",
                    "func": "lambda x, y: min(1.0, max(0, 1.0 - norm.pdf((x - {1}) * {0}) * norm.pdf((y - {3}) * {2}) * 7))".format(spread, peakX, spread, peakY),
                    "param1": param1,
                    "param2": param2,
                    "weight": roundPrecision(random.uniform(0, 3))
                }
        elif type == 2:
            xScale = roundPrecision(random.uniform(0.1, 3 * math.tau))
            yScale = roundPrecision(random.uniform(0.1, 3 * math.tau))

            xPhase = roundPrecision(random.uniform(0.1, 3 * math.tau))
            yPhase = roundPrecision(random.uniform(0.1, 3 * math.tau))

            return {
                "type": "waves",
                "func": "lambda x, y: (math.sin(x*{0} + {1}) + 1.0) * (math.sin(y*{2} + {3}) + 1.0) / 4.0".format(xScale, xPhase, yScale, yPhase),
                "param1": param1,
                "param2": param2,
                "weight": roundPrecision(random.uniform(0, 3))
            }
        elif type == 3:
            sizeX = random.randint(3, 6)
            sizeY = random.randint(3, 6)
            grid = []
            for x in range(sizeY):
                row = []
                for y in range(sizeX):
                    row.append(roundPrecision(random.uniform(0, 1)))
                grid.append(row)

            return {
                "type": "random",
                "func": "scipy.interpolate.interp2d({0}, {1}, {2}, kind='linear')".format(json.dumps(list(numpy.linspace(0, 1.0, sizeX))), json.dumps(list(numpy.linspace(0, 1.0, sizeY))), json.dumps(grid)),
                "param1": param1,
                "param2": param2,
                "weight": roundPrecision(random.uniform(0, 3))
            }

    def createHyperParameterContribution(self, param, type = None):
        if type is None:
            type = random.randint(0, 4)
        if type == 0:
            xStart = roundPrecision(random.uniform(0, 1))
            xEnd = roundPrecision(random.uniform(0, 1))
            xSlope = roundPrecision(xEnd - xStart)

            return {
                "type": "linear",
                "func": "lambda x: ({0} + {1} * x)".format(xStart, xSlope),
                "param": param,
                "weight": roundPrecision(random.uniform(0, 3))
            }
        elif type == 1:
            optimalPoint = roundPrecision(random.uniform(0, 1))

            invert = random.choice([True, False])

            if invert:
                return {
                    "type": "hill",
                    "func": "lambda x: min(1.0, max(0, 1.0 - ( math.sin(x*3.14 - {0}) / 2.0 + 0.5 ) ))".format(optimalPoint),
                    "param": param,
                    "weight": roundPrecision(random.uniform(0, 3))
                }
            else:
                return {
                    "type": "hill",
                    "func": "lambda x: min(1.0, max(0, ( math.sin(x*3.14 - {0}) / 2.0 + 0.5 )))".format(optimalPoint),
                    "param": param,
                    "weight": roundPrecision(random.uniform(0, 3))
                }
        elif type == 2:
            invert = random.choice([True, False])

            height = roundPrecision(random.uniform(0, 0.3))

            if invert:
                return {
                    "type": "exponential",
                    "func": "lambda x: min(1.0, max(0, 1.0 - (0.1 * math.pow(10, x) + {0})))".format(height),
                    "param": param,
                    "weight": roundPrecision(random.uniform(0, 3))
                }
            else:
                return {
                    "type": "exponential",
                    "func": "lambda x: min(1.0, max(0, 0.1 * (math.pow(10, x) + {0})))".format(height),
                    "param": param,
                    "weight": roundPrecision(random.uniform(0, 3))
                }
        elif type == 3:
            invert = random.choice([True, False])

            if invert:
                return {
                    "type": "logarithmic",
                    "func": "lambda x: min(1.0, max(0, 1.0 - (1.0 - math.log10(9*x+1))))",
                    "param": param,
                    "weight": roundPrecision(random.uniform(0, 3))
                }
            else:
                return {
                    "type": "logarithmic",
                    "func": "lambda x: min(1.0, max(0, (1.0 - math.log10(9*x+1))))",
                    "param": param,
                    "weight": roundPrecision(random.uniform(0, 3))
                }
        elif type == 4:
            # Random
            sizeX = random.randint(3, 8)
            values = [roundPrecision(random.uniform(0, 1)) for n in range(sizeX)]

            return {
                "type": "random",
                "func": "scipy.interpolate.interp1d({0}, {1})".format(json.dumps(list(numpy.linspace(0, 1, sizeX))), json.dumps(values)),
                "param": param,
                "weight": roundPrecision(random.uniform(0, 3))
            }


    def createSearchFunction(self):
        parameters = [self.createHyperParameter() for n in range(5)]

        probabilityOfInteraction = 0.3

        contributions = []
        for parameter in parameters:
            contributions.append(self.createHyperParameterContribution(parameter))

        interactions = []
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
                    computeScript += "    {0}_loss += interactions[{1}](params[\"{2}\"], params[\"{3}\"]) * {4}\n".format(parameter['name'], str(index), interaction['param1']['name'], interaction['param2']['name'], interaction['weight'])
                    interactionsWeight += interaction['weight']
                    # computeScript += "    print(\"interactions[{0}]\", interactions[{1}](params[\"{2}\"], params[\"{3}\"]) * {4})\n".format(str(index), str(index), interaction['param1']['name'], interaction['param2']['name'], interaction['weight'])
            contributionWeight = random.uniform(0.1, 0.4)
            computeScript += "    loss += {0}_loss * {1}\n".format(parameter['name'], parameter['weight'] / (interactionsWeight if interactionsWeight > 0 else 1.0) * (1.0 - contributionWeight))
            computeScript += "    loss += {0}_contribution * {1}\n".format(parameter['name'], parameter['weight'] / (contributions[parameterIndex]['weight']) * contributionWeight)
            totalParameterWeight += parameter['weight']

        computeScript += "    loss /= {0}\n".format(totalParameterWeight)
        # computeScript += "    print(loss)\n".format(totalParameterWeight)
        computeScript += "    return {\"loss\":float(loss[0]) if not isinstance(loss, float) else loss, \"status\": \"ok\"}\n"

        with open("test.py", 'wt') as file:
            file.write(computeScript)

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

        with open('search.json', 'wt') as file:
            json.dump(search, file, indent=4)

    def run(self):
        self.createSearchFunction()
        testGlobals = {}
        with open('test.py', 'rt') as file:
            exec(file.read(), testGlobals)

        computeLoss = testGlobals['computeLoss']

        space = {}
        with open('search.json', 'rt') as file:
            space = json.load(file)

        # Perform a random search to establish roughly an optimal point
        best = hyperopt.fmin(fn=computeLoss,
                      algo=hyperopt.rand.suggest,
                      space={paramName: hyperopt.hp.uniform(paramName, 0, 1) for paramName in space['hyperparameters']['properties']},
                      max_evals=1000)
        benchmarkScore = computeLoss(best)['loss']

        averageLengths = []
        for n in range(10):
        # Now we run TPE and see how long it takes to beat the value found by random search, on average
            trials = hyperopt.Trials()
            bestScore = 1.0
            count = 0
            while bestScore > benchmarkScore and count < 5000:
                best = hyperopt.fmin(fn=computeLoss,
                                     algo=hyperopt.rand.suggest,
                                     space={paramName: hyperopt.hp.uniform(paramName, 0, 1) for paramName in space['hyperparameters']['properties']},
                                     max_evals=count + 5,
                                     trials=trials)
                bestScore = computeLoss(best)['loss']
                count += 5
            averageLengths.append(count)

        print(averageLengths)


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
    exec("func = " + interaction['func'], funcStore)
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
    exec("import scipy.interpolate\nfunc = " + contribution['func'], funcStore)
    func = funcStore['func']

    xVals = numpy.linspace(0, 1, 25)

    yVals = []
    for x in xVals:
        yVals.append(func(x))

    # Plot the surface.
    surf = ax.scatter(numpy.array(xVals), numpy.array(yVals), cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=0, vmax=1)

    plt.show()


if __name__ == '__main__':
    # createContributionChartExample()

    algo = AlgorithmSimulation()
    algo.run()