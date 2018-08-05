import jsonschema
import os
import os.path
import math
import csv
from hypermax.hyperparameter import Hyperparameter
import sklearn.covariance
import numpy
from hypermax.utils import roundPrecision
import matplotlib
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
import concurrent.futures
import colors

class ResultsAnalyzer:
    """
        This class is responsible for analyzing the results and outputting various types of analysis to the results folder.
    """

    def __init__(self, config):
        resultsConfig = config.get('results', {"results_directory": "results"})

        jsonschema.validate(resultsConfig, self.configurationSchema())

        self.config = config
        self.resultsConfig = resultsConfig
        self.parameters = None

        # Determine if the results directory exists already. If so, we add a suffix
        increment = 0
        while os.path.exists(resultsConfig['results_directory'] + "_" + str(increment)):
            increment += 1
        self.directory = resultsConfig['results_directory'] + "_" + str(increment)

        self.fig = None

    @classmethod
    def configurationSchema(self):
        """ This method returns the configuration schema for the results formatter module. The schema
            is a standard JSON-schema object."""
        return {
            "type": "object",
            "properties": {
                "results_directory": {"type": "string"}
            }
        }

    def makeDirs(self, dir):
        """ Recursively ensures all the directories leading up to dir actually exist."""
        if not os.path.exists(dir):
            parts = []
            while dir:
                dir, tail = os.path.split(dir)
                parts.insert(0, tail)
            for index in range(len(parts)):
                if index > 0:
                    partialDirectory = os.path.join(*parts[:index+1])
                else:
                    partialDirectory = parts[0]

                if not os.path.exists(partialDirectory):
                    os.mkdir(partialDirectory)

    def generateExports(self, results, parameter1, parameter2):
        lossCsvFilename = 'loss_matrix_' + parameter1.root[5:] + '_' + parameter2.root[5:] + '.csv'
        lossImageFilename = 'loss_matrix_' + parameter1.root[5:] + '_' + parameter2.root[5:] + '.png'
        subDirectory = os.path.join(self.directory, parameter1.root[5:], parameter2.root[5:])
        self.makeDirs(subDirectory)
        self.exportLossMatrixToCSV(os.path.join(subDirectory, lossCsvFilename), results, parameter1, parameter2, 'loss')
        self.exportLossMatrixToImage(os.path.join(subDirectory, lossImageFilename), results, parameter1, parameter2, 'loss', 'Loss Matrix')

        timeCsvFilename = 'time_matrix_' + parameter1.root[5:] + '_' + parameter2.root[5:] + '.csv'
        timeImageFilename = 'time_matrix_' + parameter1.root[5:] + '_' + parameter2.root[5:] + '.png'

        self.exportLossMatrixToCSV(os.path.join(subDirectory, timeCsvFilename), results, parameter1, parameter2, 'time')
        self.exportLossMatrixToImage(os.path.join(subDirectory, timeImageFilename), results, parameter1, parameter2, 'time', 'Time Matrix')

    def outputResultsFolder(self, optimizer, detailed=True):
        # Ensure the directory we want to store results in is there
        self.makeDirs(self.directory)

        resultsFile = os.path.join(self.directory, 'results.csv')
        self.exportResultsCSV(resultsFile, optimizer)

        correlationsFile = os.path.join(self.directory, 'correlations.csv')
        self.exportCorrelationsToCSV(correlationsFile, optimizer)

        # Only do these results if detailed is enabled, since they take a lot more computation
        if detailed:
            parameters = Hyperparameter(optimizer.config.data['hyperparameters']).getFlatParameters()

            # We use concurrent.futures.ProcessThreadPool here for two reasons. One for speed (since generating the images can be slow)
            # The other is because  matplotlib is not inherently thread safe.
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                for parameter1 in parameters:
                    if parameter1.config['type'] == 'number':
                        for parameter2 in parameters:
                            if parameter2.config['type'] == 'number':
                                if parameter1.root != parameter2.root:
                                    # self.generateExports(list(optimizer.results), parameter1, parameter2)
                                    executor.submit(self.generateExports, list(optimizer.results), parameter1, parameter2)

    def exportResultsCSV(self, fileName, optimizer):
        with open(fileName, 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=optimizer.results[0].keys(), dialect='unix')
            writer.writeheader()
            writer.writerows(optimizer.results)


    def exportCorrelationsToCSV(self, fileName, optimizer):
        matrix, labels = self.computeCorrelations(optimizer)

        data = []
        for index, row in enumerate(matrix):
            rowData = {
                'field': labels[index]
            }

            for labelIndex, label in enumerate(labels):
                rowData[label] = row[labelIndex]

            data.append(rowData)

        with open(fileName, 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=['field'] + labels)
            writer.writerows(data)


    def exportLossMatrixToCSV(self, fileName, results, parameter1, parameter2, valueKey='loss'):
        scores, parameter1Buckets, parameter2Buckets = self.computeLossMatrix(results, parameter1, parameter2, valueKey)

        with open(fileName, 'wt') as file:
            writer = csv.writer(file)
            param1Padding = int((len(parameter1Buckets)-1)/2)-1
            param2Padding = int((len(parameter2Buckets)-1)/2)

            writer.writerow([''] + ([''] * param2Padding) + [parameter2.root[5:]] + ([''] * (param2Padding)))
            writer.writerow(['',''] + parameter2Buckets)
            for rowIndex, row in enumerate(scores):
                if rowIndex == param1Padding:
                    writer.writerow([parameter1.root[5:], str(parameter1Buckets[rowIndex])] + row)
                else:
                    writer.writerow(['', str(parameter1Buckets[rowIndex])] + row)


    def exportLossMatrixToImage(self, fileName, results, parameter1, parameter2, valueKey='loss', title='Loss Matrix'):
        scores, parameter1Buckets, parameter2Buckets = self.computeLossMatrix(results, parameter1, parameter2, valueKey)

        minVal = float(numpy.min(scores))
        maxVal = float(numpy.max(scores))
        yellowVal = float(numpy.percentile(scores, q=30))
        greenVal = float(numpy.percentile(scores, q=15))

        green = numpy.array(colors.rgb(0, 1, 0).hsv._color)
        yellow = numpy.array(colors.rgb(1, 1, 0).hsv._color)
        red = numpy.array(colors.rgb(1, 0, 0).hsv._color)
        blue = numpy.array(colors.rgb(0.3, 0.3, 1).hsv._color)

        colorGrid = []
        for row in scores:
            colorRow = []
            for score in row:
                if score <= greenVal:
                    valRange = max(0.1, (greenVal - minVal))
                    dist = (score - minVal) / valRange
                    color = colors.hsv(*(green * dist + blue * (1.0 - dist)))
                elif score <= yellowVal:
                    valRange = max(0.1, (yellowVal - greenVal))
                    dist = (score - greenVal) / valRange
                    color = colors.hsv(*(yellow * dist + green * (1.0 - dist)))
                else:
                    valRange = max(0.1, (maxVal - yellowVal))
                    dist = (score - yellowVal) / valRange
                    color = colors.hsv(*(red * dist + yellow * (1.0 - dist)))
                colorRow.append(color.rgb._color)
            colorGrid.append(colorRow)

        fig, ax = plt.subplots()

        im = ax.imshow(numpy.array(colorGrid, dtype=numpy.float32), interpolation='bicubic')

        # We want to show all ticks...
        ax.set_xticks(numpy.arange(len(parameter2Buckets)))
        ax.set_yticks(numpy.arange(len(parameter1Buckets)))

        # ... and label them with the respective list entries
        ax.set_xticklabels(parameter2Buckets)
        ax.set_yticklabels(parameter1Buckets)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Function which formats the text for display in each cell of heatmap
        def getText(i,j):
            cellText = str(roundPrecision(scores[i][j], 2))
            if cellText[:2] == '0.':
                cellText = cellText[1:]  # Eliminate the 0 from 0.xx
            return cellText

        # Determine the longest string we have to put into the heatmap
        longest = 0
        for i in range(len(parameter1Buckets)):
            for j in range(len(parameter2Buckets)):
                longest = max(len(getText(i, j)), longest)

        # Loop over data dimensions and create text annotations.
        fontSize = int(12 - max(0, 1.25*(longest - 3)))
        for i in range(len(parameter1Buckets)):
            for j in range(len(parameter2Buckets)):
                ax.text(j, i, getText(i,j), ha="center", va="center", color="black", fontsize=fontSize)

        ax.set_title(title + " of " + parameter1.root[5:] + " vs " + parameter2.root[5:])
        fig.tight_layout()
        plt.savefig(fileName, dpi=200)
        plt.close()


    def computeCorrelations(self, optimizer):
        keys = Hyperparameter(optimizer.config.data['hyperparameters']).getFlatParameterNames()

        values = {}
        types = {}
        for key in keys:
            values[key] = set()
            types[key] = set()

        for result in optimizer.results:
            for key in keys:
                value = result[key[5:]]
                values[key].add(value)
                types[key].add(type(value).__name__)

        vectors = []
        labels = []
        outputs = []
        for result in optimizer.results:
            vector = []
            vectorLabels = []
            for key in keys:
                value = result[key[5:]]
                if 'bool' in types[key] or 'int' in types[key] or 'float' in types[key]:
                    if isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                        vector.append(float(value))
                        vectorLabels.append(key + ".number")
                    else:
                        vector.append(-1)
                        vectorLabels.append(key + ".number")
                if 'NoneType' in types[key]:
                    if value is None:
                        vector.append(value)
                        vectorLabels.append(key + ".none")
                    else:
                        vector.append(-1)
                        vectorLabels.append(key + ".none")
                if 'str' in types[key]:
                    classes = [v for v in values[key] if isinstance(v, str)]

                    if isinstance(value, str):
                        for v in classes:
                            if value == v:
                                vector.append(1.0)
                            else:
                                vector.append(0.0)
                            vectorLabels.append(key + ".class." + v)
                    else:
                        for v in classes:
                            vector.append(0)
                            vectorLabels.append(key + ".class." + v)
            vectors.append(vector)
            outputs.append(result['loss'])
            if not labels:
                labels = vectorLabels

        model = sklearn.covariance.LedoitWolf()
        model.fit(numpy.array(vectors), numpy.array(outputs))

        covariances = model.covariance_
        correlations = numpy.zeros_like(covariances)

        deviations = numpy.std(vectors, axis=0)

        for label1Index in range(len(labels)):
            for label2Index in range(len(labels)):
                correlations[label1Index][label2Index] = covariances[label1Index][label2Index] / (deviations[label1Index] * deviations[label2Index])

        return correlations, labels

    def computeLossMatrix(self, results, parameter1, parameter2, valueKey='loss'):
        """
            This computes the loss matrix between two hyper-parameters. The loss matrix
            helps you to visualize what are the best areas of the hyper parameter space
            by plotting them on a grid and coloring them.
        """
        # Divide the range up into 100 parts
        numBuckets = 10
        parameter1Buckets = []
        if parameter1.config['scaling'] == 'linear':
            domain = parameter1.config['max'] - parameter1.config['min']
            parameter1Buckets = list(numpy.arange(parameter1.config['min'], parameter1.config['max'] + (domain / numBuckets), domain / numBuckets))[1:]
        elif parameter1.config['scaling'] == 'logarithmic':
            logMax = math.log(parameter1.config['max'])
            logMin = math.log(parameter1.config['min'])
            domain = logMax - logMin
            logBuckets = numpy.arange(logMin, logMax + (domain / numBuckets), domain / numBuckets)[1:]
            parameter1Buckets = [math.exp(n) for n in logBuckets]

        parameter2Buckets = []
        if parameter2.config['scaling'] == 'linear':
            domain = parameter2.config['max'] - parameter2.config['min']
            parameter2Buckets = list(numpy.arange(parameter2.config['min'], parameter2.config['max'] + (domain / numBuckets), domain / numBuckets))[1:]
        elif parameter2.config['scaling'] == 'logarithmic':
            logMax = math.log(parameter2.config['max'])
            logMin = math.log(parameter2.config['min'])
            domain = logMax - logMin
            logBuckets = numpy.arange(logMin, logMax + (domain / numBuckets), domain / numBuckets)[1:]
            parameter2Buckets = [math.exp(n) for n in logBuckets]

        # Round the precision for each of the buckets
        parameter1Buckets = [roundPrecision(value) for value in parameter1Buckets]
        parameter2Buckets = [roundPrecision(value) for value in parameter2Buckets]

        # Create a grid for each of the values
        resultGrid = []
        for value in parameter1Buckets:
            row = []
            for value in parameter2Buckets:
                row.append([])
            resultGrid.append(row)

        # Go through each of the results, and put them into one of the buckets (or exclude them if these hyper-parameters were not active
        for result in results:
            paramater1Key = parameter1.root[5:]
            paramater2Key = parameter2.root[5:]

            if isinstance(result[paramater1Key], float) or isinstance(result[paramater1Key], int) or isinstance(result[paramater1Key], bool):
                if isinstance(result[paramater2Key], float) or isinstance(result[paramater2Key], int) or isinstance(result[paramater2Key], bool):
                    parameter1Value = float(result[paramater1Key])
                    parameter2Value = float(result[paramater2Key])

                    parameter1Index = None
                    for index1, value1 in enumerate(parameter1Buckets):
                        if parameter1Value < value1:
                            parameter1Index = index1
                            break
                    parameter2Index = None
                    for index2, value2 in enumerate(parameter2Buckets):
                        if parameter2Value < value2:
                            parameter2Index = index2
                            break
                    resultGrid[parameter1Index][parameter2Index].append(result)

        # Now go through each entry in the grid and compute the average score
        scoreGrid = []
        for row in resultGrid:
            scoreRow = []
            for column in row:
                if len(column) > 0:
                    scoreRow.append(numpy.min([result[valueKey] for result in column]))
                else:
                    scoreRow.append(None)
            scoreGrid.append(scoreRow)

        # Now for any parts of the grid which don't have a score, we find the nearest neighbor on the grid and just take its score. If theres multiple, we take the average of them.
        for rowIndex, row in enumerate(scoreGrid):
            for columnIndex, column in enumerate(row):
                if column is None:
                    closest = None
                    closestValues = []
                    # Find the nearest location on the score grid which has a value
                    for possibleRowIndex, possibleRow in enumerate(scoreGrid):
                        for possibleColumnIndex, possibleColumn in enumerate(possibleRow):
                            if possibleColumn is not None:
                                dist = (rowIndex - possibleRowIndex) * (rowIndex - possibleRowIndex) + (columnIndex - possibleColumnIndex) * (columnIndex - possibleColumnIndex)
                                if closest is None or dist < closest:
                                    closest = dist
                                    closestValues = [possibleColumn]
                                elif dist == closest:
                                    closestValues.append(possibleColumn)
                    # Take the mean of the closest values and use that for this location on the score grid
                    scoreGrid[rowIndex][columnIndex] = numpy.mean(closestValues)

        # Round all of the values in the final score grid. This makes them more pleasant to look at for display purposes.
        for rowIndex, row in enumerate(scoreGrid):
            for columnIndex, column in enumerate(row):
                scoreGrid[rowIndex][columnIndex] = roundPrecision(scoreGrid[rowIndex][columnIndex])

        return scoreGrid, parameter1Buckets, parameter2Buckets




