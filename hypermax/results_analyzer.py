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
import matplotlib.cm
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
from pprint import pprint
import concurrent.futures
import colors
import traceback
import functools
import warnings
import scipy.optimize
import random
import json
import atexit
import matplotlib.ticker as mticker
import traceback


def handleChartException(function):
    """
        A decorator that wraps the given chart-generation function and handles and exceptions that might have been thrown during.

        Said exceptions are ignored except when developing since they usually come from poor data that isn't useful to plot anyhow
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            plt.close('all')
            # traceback.print_exc()
            # raise # reraise the exception and allow it to bubble so developers can catch why the charts aren't being generated.

    return wrapper


class ResultsAnalyzer:
    """
        This class is responsible for analyzing the results and outputting various types of analysis to the results folder.
    """

    def __init__(self, config):
        resultsConfig = config.get('results', {"directory": "results"})

        jsonschema.validate(resultsConfig, self.configurationSchema())

        self.config = config
        self.resultsConfig = resultsConfig
        self.parameters = None

        # Determine if the results directory exists already. If so, we add a suffix
        increment = 0
        while os.path.exists(resultsConfig.get('directory', 'results') + "_" + str(increment)):
            increment += 1
        self.directory = resultsConfig.get('directory', 'results') + "_" + str(increment)

        self.fig = None
        self.completedCharts = 0
        self.totalCharts = 0

        self.singleParameterLossFigure = None
        self.twoParameterScatterFigure = None
        self.twoParameterLossFigure = None

        self.singleParameterLossAxes = None
        self.twoParameterScatterAxes = None
        self.twoParameterLossAxes = None

    def __del__(self):
        if self.singleParameterLossFigure:
            plt.close(self.singleParameterLossFigure)
        elif self.twoParameterLossFigure:
            plt.close(self.twoParameterLossFigure)
        elif self.twoParameterScatterFigure:
            plt.close(self.twoParameterScatterFigure)

    @classmethod
    def configurationSchema(self):
        """ This method returns the configuration schema for the results formatter module. The schema
            is a standard JSON-schema object."""
        return {
            "type": "object",
            "properties": {
                "graphs": {"type": "boolean"},
                "directory": {"type": "string"}
            }
        }

    def makeDirs(self, dir):
        """ Recursively ensures all the directories leading up to dir actually exist."""
        if not os.path.exists(dir):
            parts = []
            while dir:
                dir, tail = os.path.split(dir)
                parts.insert(0, tail)
                if dir == '/':
                    parts.insert(0, '/')
                    break
            for index in range(len(parts)):
                if index > 0:
                    partialDirectory = os.path.join(*parts[:index + 1])
                else:
                    partialDirectory = parts[0]

                if not os.path.exists(partialDirectory):
                    os.mkdir(partialDirectory)

    def generateMultiParameterExports(self, results, parameter1, parameter2, directory=None):
        try:
            if directory is None:
                directory = self.directory

            subDirectory = os.path.join(directory, parameter1.root[5:], parameter2.root[5:])
            self.makeDirs(subDirectory)

            for reduction in ['mean', 'min', 'median']:
                lossCsvFilename = 'loss_matrix_' + reduction + "_" + parameter1.root[5:] + '_' + parameter2.root[5:] + '.csv'
                lossImageFilename = 'loss_matrix_' + reduction + "_" + parameter1.root[5:] + '_' + parameter2.root[5:] + '.png'
                lossCsvTop10PercentFilename = 'loss_matrix_top_10_percent_' + reduction + "_" + parameter1.root[5:] + '_' + parameter2.root[5:] + '.csv'
                lossImageTop10PercentFilename = 'loss_matrix_top_10_percent_' + reduction + "_" + parameter1.root[5:] + '_' + parameter2.root[5:] + '.png'
                responseImageFilename = 'response_matrix_' + reduction + "_" + parameter1.root[5:] + '_' + parameter2.root[5:] + '.png'
                responseImageTop10PercentFilename = 'response_matrix_top_10_percent_' + reduction + "_" + parameter1.root[5:] + '_' + parameter2.root[5:] + '.png'
                self.exportLossMatrixToCSV(os.path.join(subDirectory, lossCsvFilename), results, parameter1, parameter2, 'loss', cutoff=1.0, reduction=reduction)
                self.exportLossMatrixToImage(os.path.join(subDirectory, lossImageFilename), results, parameter1, parameter2, 'loss', 'Loss Matrix (' + reduction + ")", cutoff=1.0,
                                             reduction=reduction)
                self.exportLossMatrixToImage(os.path.join(subDirectory, responseImageFilename), results, parameter1, parameter2, 'loss', 'Response Matrix (' + reduction + ")",
                                             cutoff=1.0, mode='response', reduction=reduction)
                self.exportLossMatrixToCSV(os.path.join(subDirectory, lossCsvTop10PercentFilename), results, parameter1, parameter2, 'loss', cutoff=0.1, reduction=reduction)
                self.exportLossMatrixToImage(os.path.join(subDirectory, lossImageTop10PercentFilename), results, parameter1, parameter2, 'loss',
                                             'Loss Matrix (top 10 percent, ' + reduction + ')', cutoff=0.1, reduction=reduction)
                self.exportLossMatrixToImage(os.path.join(subDirectory, responseImageTop10PercentFilename), results, parameter1, parameter2, 'loss',
                                             'Response Matrix (top 10 percent, ' + reduction + ')', cutoff=0.1, mode='response', reduction=reduction)

            scatterFilename = 'scatter_' + parameter1.root[5:] + '_' + parameter2.root[5:] + '.png'
            scatterTop10PercentFilename = 'scatter_top_10_percent' + parameter1.root[5:] + '_' + parameter2.root[5:] + '.png'
            self.exportTwoParameterScatter(os.path.join(subDirectory, scatterFilename), results, parameter1, parameter2, 'loss', cutoff=1.0, title='Scatter Chart')
            self.exportTwoParameterScatter(os.path.join(subDirectory, scatterTop10PercentFilename), results, parameter1, parameter2, 'loss', cutoff=0.1,
                                           title='Scatter Chart (top 10 percent)')

            timeCsvFilename = 'time_matrix_' + parameter1.root[5:] + '_' + parameter2.root[5:] + '.csv'
            timeImageFilename = 'time_matrix_' + parameter1.root[5:] + '_' + parameter2.root[5:] + '.png'

            self.exportLossMatrixToCSV(os.path.join(subDirectory, timeCsvFilename), results, parameter1, parameter2, 'time')
            self.exportLossMatrixToImage(os.path.join(subDirectory, timeImageFilename), results, parameter1, parameter2, 'time', 'Time Matrix')
        except Exception as e:
            traceback.print_exc()
            return e

    def generateSingleParameterExports(self, results, parameter, directory=None):
        try:
            if directory is None:
                directory = self.directory

            subDirectory = os.path.join(directory, parameter.root[5:])
            self.makeDirs(subDirectory)

            lossCsvFilename = 'losses_' + parameter.root[5:] + '.csv'
            lossImageFilename = 'loss_chart_' + parameter.root[5:] + '.png'
            lossTop10ImageFilename = 'loss_chart_top_10_percent_' + parameter.root[5:] + '.png'
            self.exportSingleParameterLossCSV(os.path.join(subDirectory, lossCsvFilename), results, parameter, 'loss')
            self.exportSingleParameterLossChart(os.path.join(subDirectory, lossImageFilename), results, parameter, 'loss', 'Loss Chart')
            self.exportSingleParameterLossChart(os.path.join(subDirectory, lossTop10ImageFilename), results, parameter, 'loss', 'Loss Chart (top 10%)', cutoff=0.1)

            for reduction in ['mean', 'min', 'median']:
                lossBucketedCsvFilename = 'losses_' + reduction + '_bucketed_' + parameter.root[5:] + '.csv'
                lossBucketedImageFilename = 'loss_chart_'+reduction+'_bucketed_' + parameter.root[5:] + '.png'
                lossTop10BucketedImageFilename = 'loss_chart_top_10_percent_'+reduction+'_bucketed_' + parameter.root[5:] + '.png'

                self.exportSingleParameterLossCSV(os.path.join(subDirectory, lossBucketedCsvFilename), results, parameter, 'loss', numBuckets=20, reduction=reduction)
                self.exportSingleParameterLossChart(os.path.join(subDirectory, lossBucketedImageFilename), results, parameter, 'loss', 'Loss Chart (20 Buckets, ' + reduction + ')', numBuckets=20, reduction=reduction)
                self.exportSingleParameterLossChart(os.path.join(subDirectory, lossTop10BucketedImageFilename), results, parameter, 'loss', 'Loss Chart (20 buckets, ' + reduction + ', top 10%)', cutoff=0.1, numBuckets=20, reduction=reduction)


            timeCsvFilename = 'times_' + parameter.root[5:] + '.csv'
            timeBucketedCsvFilename = 'times_bucketed_' + parameter.root[5:] + '.csv'
            timeImageFilename = 'time_chart_' + parameter.root[5:] + '.png'
            timeTop10ImageFilename = 'time_chart_top_10_percent_' + parameter.root[5:] + '.png'

            self.exportSingleParameterLossCSV(os.path.join(subDirectory, timeCsvFilename), results, parameter, 'time')
            self.exportSingleParameterLossCSV(os.path.join(subDirectory, timeBucketedCsvFilename), results, parameter, 'time', numBuckets=20)
            self.exportSingleParameterLossChart(os.path.join(subDirectory, timeImageFilename), results, parameter, 'time', 'Time Chart', numBuckets=20)
            self.exportSingleParameterLossChart(os.path.join(subDirectory, timeTop10ImageFilename), results, parameter, 'time', 'Time Chart (top 10%)', cutoff=0.1)
        except Exception as e:
            traceback.print_exc()
            return e

    def outputResultsFolder(self, optimizer, detailed=True, workers=1, directory=None):
        parameters = Hyperparameter(optimizer.config.data['hyperparameters']).getFlatParameters()

        if directory is None:
            directory = self.directory

        # Ensure the directory we want to store results in is there
        self.makeDirs(directory)

        resultsFile = os.path.join(directory, 'results.csv')
        optimizer.exportResultsCSV(resultsFile)
        guidanceFile = os.path.join(directory, 'guidance.json')
        optimizer.exportGuidanceJSON(guidanceFile)

        with open(os.path.join(directory, 'search.json'), 'wt') as paramsFile:
            json.dump(optimizer.config.data, paramsFile, indent=4, sort_keys=True)

        if optimizer.best:
            with open(os.path.join(directory, 'best.json'), 'wt') as paramsFile:
                json.dump(optimizer.best, paramsFile, indent=4, sort_keys=True)

        with open(os.path.join(directory, 'current_trials.json'), 'wt') as paramsFile:
            trials = []
            for trial in optimizer.currentTrials:
                trialData = {key: value for key,value in trial.items() if key != 'start'}
                trials.append(trialData)
            trials = list(sorted(trials, key=lambda trial: trial['worker']))
            json.dump(trials, paramsFile, indent=4, sort_keys=True)

        def onChartCompleted(e):
            self.completedCharts += 1

        # Only do these results if detailed is enabled, since they take a lot more computation
        if len(optimizer.results) > 5 and detailed:
            correlationsFile = os.path.join(directory, 'correlations.csv')
            self.exportCorrelationsToCSV(correlationsFile, optimizer)

            self.completedCharts = 0
            self.totalCharts = 0
            self.totalCharts = len(parameters) + (len(parameters) - 1) * (len(parameters) - 1)

            # We use concurrent.futures.ProcessThreadPool here for two reasons. One for speed (since generating the images can be slow)
            # The other is because  matplotlib is not inherently thread safe.
            futures = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for parameter1 in parameters:
                    # self.generateSingleParameterExports(list(optimizer.results), parameter1)
                    if parameter1.config['type'] == 'number':
                        futures.append(executor.submit(self.generateSingleParameterExports, list(optimizer.results), parameter1, directory))
                        for parameter2 in parameters:
                            if parameter2.config['type'] == 'number':
                                if parameter1.root != parameter2.root:
                                    # self.generateMultiParameterExports(list(optimizer.results), parameter1, parameter2)
                                    futures.append(executor.submit(self.generateMultiParameterExports, list(optimizer.results), parameter1, parameter2, directory))
                for future in futures:
                    future.add_done_callback(onChartCompleted)
                    def shutdown():
                        future.cancel()
                    atexit.register(shutdown)
                    future.add_done_callback(lambda a: atexit.unregister(shutdown))

            for future in futures:
                if future.result() is not None:
                    print(traceback.format_exception_only(Exception, future.result()))
                    raise future.result()

    @handleChartException
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
            writer.writeheader()
            writer.writerows(data)

    @handleChartException
    def exportTwoParameterScatter(self, fileName, results, parameter1, parameter2, valueKey='loss', cutoff=1.0, title="Scatter"):
        parameter1Key = parameter1.root[5:]
        parameter2Key = parameter2.root[5:]

        losses = [result[valueKey] for result in results if result[valueKey] is not None]
        threshhold = numpy.percentile(losses, cutoff * 100)

        # Put all of the results into buckets based on their value for parameter1 / parameter2
        resultsMerged = {}
        for result in results:
            if result[valueKey] is not None and result[valueKey] <= threshhold:
                if isinstance(result[parameter1Key], float) or isinstance(result[parameter1Key], int) or isinstance(result[parameter1Key], bool):
                    if isinstance(result[parameter2Key], float) or isinstance(result[parameter2Key], int) or isinstance(result[parameter2Key], bool):
                        key = (result[parameter1Key], result[parameter2Key])

                        if key not in resultsMerged:
                            resultsMerged[key] = []

                        resultsMerged[key].append(result[valueKey])

        losses = [numpy.min(vals) for key,vals in resultsMerged.items()]

        xCoords = []
        yCoords = []
        scores = []
        sizes = []

        minVal = float(numpy.min(losses))
        maxVal = float(numpy.max(losses))
        redVal = float(numpy.percentile(losses, q=90))
        yellowVal = float(numpy.percentile(losses, q=40))
        greenVal = float(numpy.percentile(losses, q=20))
        blueVal = float(numpy.percentile(losses, q=5))

        if (maxVal - minVal) > 0:
            redVal = (redVal - minVal) / (maxVal - minVal)
            yellowVal = (yellowVal - minVal) / (maxVal - minVal)
            greenVal = (greenVal - minVal) / (maxVal - minVal)
            blueVal = (blueVal - minVal) / (maxVal - minVal)
        else:
            redVal = 0.9
            yellowVal = 0.6
            greenVal = 0.3
            blueVal = 0.1

        green = numpy.array([0, 1, 0, 1])
        yellow = numpy.array([1, 1, 0, 1])
        red = numpy.array([1, 0, 0, 1])
        blue = numpy.array([0.3, 0.3, 1, 1])

        if self.twoParameterScatterFigure is None:
            fig, ax = plt.subplots()
            self.twoParameterScatterFigure = fig
            self.twoParameterScatterAxes = ax
        else:
            fig = self.twoParameterScatterFigure
            ax = self.twoParameterScatterAxes
            ax.clear()

        # We shuffle the results here because the scatter plot will by default draw later points over earlier ones, and later results will be on average better
        # anyhow, skewing the graphic
        resultsShuffled = list(results)
        random.shuffle(resultsShuffled)
        for key, resultScores in resultsMerged.items():
            xCoords.append(key[0])
            yCoords.append(key[1])
            if (maxVal - minVal) > 0:
                value = float((numpy.min(resultScores) - minVal) / (maxVal - minVal))
                scores.append(value)
                sizeAdjust = 1.0 - min(1, ((value*value)) / (redVal*redVal))
                sizes.append(10 + sizeAdjust * 25)
            else:
                value = float((numpy.min(resultScores) - minVal))
                scores.append(value)
                sizes.append(20.0)

        if parameter1.config.get('scaling', 'linear') == 'logarithmic':
            ax.set_xscale('log')
        else:
            ax.set_xscale('linear')

        if parameter2.config.get('scaling', 'linear') == 'logarithmic':
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')

        minVal = parameter1.config.get('min')
        maxVal = parameter1.config.get('max')
        if (minVal > 0.001 and minVal < 10000 and maxVal > 0.001 and maxVal < 10000):
            ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        else:
            ax.xaxis.set_minor_formatter(mticker.LogFormatterSciNotation())
            ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())

        minVal = parameter2.config.get('min')
        maxVal = parameter2.config.get('max')
        if (minVal > 0.001 and minVal < 10000 and maxVal > 0.001 and maxVal < 10000):
            ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        else:
            ax.yaxis.set_minor_formatter(mticker.LogFormatterSciNotation())
            ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

        colorList = [(0, blue), (blueVal, blue), (greenVal, green), (yellowVal, yellow), (redVal, red), (1.0, red)]
        ax.scatter(xCoords, yCoords, c=numpy.array(scores), s=numpy.array(sizes), cmap=matplotlib.colors.LinearSegmentedColormap.from_list('loss_matrix', colorList, N=50000), alpha=0.7)

        fig.suptitle(title + " of " + parameter1.root[5:] + " vs " + parameter2.root[5:], fontdict={"fontsize": 10})

        ax.set_xlabel(parameter1Key)
        ax.set_ylabel(parameter2Key)

        fig.set_tight_layout(True)
        fig.savefig(fileName, dpi=200)
        plt.close(fig)

    @handleChartException
    def exportLossMatrixToCSV(self, fileName, results, parameter1, parameter2, valueKey='loss', cutoff=1.0, reduction='min'):
        scores, parameter1Buckets, parameter2Buckets = self.computeLossMatrix(results, parameter1, parameter2, valueKey, cutoff=cutoff, reduction=reduction)

        with open(fileName, 'wt') as file:
            writer = csv.writer(file)
            param1Padding = int((len(parameter1Buckets) - 1) / 2)
            param2Padding = int((len(parameter2Buckets) - 1) / 2) - 1

            writer.writerow(['', ''] + ([''] * param1Padding) + [parameter1.root[5:]] + ([''] * (param1Padding)) + ['', '', ''])
            writer.writerow(['', '', ''] + parameter1Buckets + ['', '', ''])
            writer.writerow(['', '', ''] + [''] * (len(parameter1Buckets)) + ['', '', ''])
            writer.writerow(['', str(parameter2Buckets[0]), ''] + [''] * (len(parameter1Buckets)) + ['', str(parameter2Buckets[0]), ''])
            for rowIndex, row in enumerate(scores):
                if rowIndex == param2Padding:
                    writer.writerow([parameter2.root[5:], str(parameter2Buckets[rowIndex + 1]), '', ''] + list(row) + ['', str(parameter2Buckets[rowIndex + 1]), parameter2.root[5:]])
                else:
                    writer.writerow(['', str(parameter2Buckets[rowIndex + 1]), '', ''] + list(row) + ['', str(parameter2Buckets[rowIndex + 1]), ''])
            writer.writerow([''] * (len(parameter1Buckets) + 3))
            writer.writerow(['', '', ''] + parameter1Buckets)
            writer.writerow(['', ''] + ([''] * param1Padding) + [parameter1.root[5:]] + ([''] * (param1Padding)))

    @handleChartException
    def exportLossMatrixToImage(self, fileName, results, parameter1, parameter2, valueKey='loss', title='Loss Matrix', cutoff=1.0, mode='global', reduction='min'):
        scores, parameter1Buckets, parameter2Buckets = self.computeLossMatrix(results, parameter1, parameter2, valueKey, cutoff=cutoff, reduction=reduction)

        minVal = float(numpy.min(scores))
        maxVal = float(numpy.max(scores))
        redVal = float(numpy.percentile(scores, q=90))
        yellowVal = float(numpy.percentile(scores, q=40))
        greenVal = float(numpy.percentile(scores, q=20))
        blueVal = float(numpy.percentile(scores, q=5))

        if mode == 'global':
            if (maxVal - minVal) > 0:
                redVal = (redVal - minVal) / (maxVal - minVal)
                yellowVal = (yellowVal - minVal) / (maxVal - minVal)
                greenVal = (greenVal - minVal) / (maxVal - minVal)
                blueVal = (blueVal - minVal) / (maxVal - minVal)
            else:
                redVal = 0.9
                yellowVal = 0.6
                greenVal = 0.3
                blueVal = 0.1
        elif mode == 'response':
            redVal = 0.9
            yellowVal = 0.6
            greenVal = 0.3
            blueVal = 0.1

        green = numpy.array([0, 1, 0])
        yellow = numpy.array([1, 1, 0])
        red = numpy.array([1, 0, 0])
        blue = numpy.array([0.3, 0.3, 1])

        colorGrid = []
        for row in scores:
            if mode == 'global':
                colorRow = []
                for score in row:
                    colorRow.append(score)
                colorGrid.append(colorRow)
            else:
                rowMinVal = float(numpy.min(row))
                rowMaxVal = float(numpy.max(row))

                if rowMaxVal == rowMinVal:
                    colorRow = [0.5] * len(row)
                    colorGrid.append(colorRow)
                else:
                    sortedRow = sorted(row)

                    colorRow = []
                    for score in row:
                        colorRow.append(float(sortedRow.index(score)) / len(row))
                    colorGrid.append(colorRow)

        if self.twoParameterLossFigure is None:
            fig, ax = plt.subplots()
            self.twoParameterLossFigure = fig
            self.twoParameterLossAxes = ax
        else:
            fig = self.twoParameterLossFigure
            ax = self.twoParameterLossAxes
            ax.clear()

        colorList = [(0, blue), (blueVal, blue), (greenVal, green), (yellowVal, yellow), (redVal, red), (1.0, red)]

        im = ax.imshow(colorGrid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('loss_matrix', colorList, N=50000), interpolation='quadric')

        # We want to show all ticks...
        ax.set_xticks(numpy.arange(len(parameter1Buckets))-0.5)
        ax.set_yticks(numpy.arange(len(parameter2Buckets))-0.5)
        #
        # if parameter1.root[5:] == 'layer_0.max_depth':
        #     print(parameter1Buckets)
        #     print(parameter2Buckets)

        # ... and label them with the respective list entries
        ax.set_xticklabels([roundPrecision(bucket) for bucket in parameter1Buckets])
        ax.set_yticklabels([roundPrecision(bucket) for bucket in parameter2Buckets])

        ax.set_xlabel(parameter1.root[5:])
        ax.set_ylabel(parameter2.root[5:])

        # Rotate the tick labels and set their alignment.
        # ax.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Function which formats the text for display in each cell of heatmap
        def getText(i, j):
            cellText = str(roundPrecision(scores[i][j], 2))
            if cellText[:2] == '0.':
                cellText = cellText[1:]  # Eliminate the 0 from 0.xx
            return cellText

        # Determine the longest string we have to put into the heatmap
        longest = 0
        for j in range(len(parameter1Buckets[1:])):
            for i in range(len(parameter2Buckets[1:])):
                longest = max(len(getText(i, j)), longest)

        # Loop over data dimensions and create text annotations.
        fontSize = int(12 - max(0, 1.25 * (longest - 3)))
        for j in range(len(parameter1Buckets[1:])):
            for i in range(len(parameter2Buckets[1:])):
                ax.text(j, i, getText(i, j), ha="center", va="center", color="black", fontsize=fontSize)

        ax.set_title(title + " of " + parameter1.root[5:] + " vs " + parameter2.root[5:], fontdict={"fontsize": 10})
        fig.set_tight_layout(True)
        fig.savefig(fileName, dpi=200)
        plt.close(fig)


    @handleChartException
    def exportSingleParameterLossChart(self, fileName, results, parameter, valueKey='loss', title='Loss Chart', cutoff=1.0, numBuckets=None, reduction='mean'):
        values, linearTrendLine, exponentialTrendLine = self.computeParameterResultValues(results, parameter, valueKey, cutoff, numBuckets, bucket_reduction=reduction)

        if self.singleParameterLossFigure is None:
            fig, ax = plt.subplots()
            self.singleParameterLossFigure = fig
            self.singleParameterLossAxes = ax
        else:
            fig = self.singleParameterLossFigure
            ax = self.singleParameterLossAxes
            ax.clear()

        fig.suptitle(title + " for " + parameter.root[5:])

        if parameter.config.get('scaling', 'linear') == 'logarithmic':
            ax.set_xscale('log')
        else:
            ax.set_xscale('linear')

        xCoords = [value[parameter.root[5:]] for value in values]
        yCoords = [value[valueKey] for value in values]

        ax.set_xlabel(parameter.root[5:])
        ax.set_ylabel(valueKey)

        ax.scatter(xCoords, yCoords)

        minVal = parameter.config.get('min')
        maxVal = parameter.config.get('max')
        if (minVal > 0.001 and minVal < 10000 and maxVal > 0.001 and maxVal < 10000):
            ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        else:
            ax.xaxis.set_minor_formatter(mticker.LogFormatterSciNotation())
            ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())

        # Preserve the limits of the scatter graph when we apply the trend line
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if linearTrendLine and exponentialTrendLine:
            trendLineXCoords = [linearTrendLine[index][0] for index in range(len(linearTrendLine))]
            ax.plot(trendLineXCoords, [(linearTrendLine[index][1], exponentialTrendLine[index][1]) for index in range(len(exponentialTrendLine))], color='red', linestyle='dashed')
        elif linearTrendLine:
            trendLineXCoords = [linearTrendLine[index][0] for index in range(len(linearTrendLine))]
            ax.plot(trendLineXCoords, [linearTrendLine[index][1] for index in range(len(linearTrendLine))], color='red', linestyle='dashed')
        elif exponentialTrendLine:
            trendLineXCoords = [exponentialTrendLine[index][0] for index in range(len(linearTrendLine))]
            ax.plot(trendLineXCoords, [exponentialTrendLine[index][1] for index in range(len(exponentialTrendLine))], color='red', linestyle='dashed')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        fig.set_tight_layout(True)
        fig.savefig(fileName, dpi=200)
        plt.close(fig)

    @handleChartException
    def exportSingleParameterLossCSV(self, fileName, results, parameter, valueKey='loss', numBuckets=None, reduction='mean'):
        newResults, linearTrendLine, exponentialTrendLine = self.computeParameterResultValues(results, parameter, valueKey, cutoff=1.0, numBuckets=numBuckets, bucket_reduction=reduction)

        with open(fileName, 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=[parameter.root[5:], valueKey, 'linearTrend', 'exponentialTrend'], dialect='unix')
            writer.writeheader()
            writer.writerows(newResults)

    def computeParameterResultValues(self, results, parameter, valueKey='loss', cutoff=1.0, numBuckets=None, bucket_reduction='mean'):
        mergedResults = {}
        for result in results:
            if (isinstance(result[parameter.root[5:]], int) or isinstance(result[parameter.root[5:]], float) or isinstance(result[parameter.root[5:]], bool)) and result[valueKey] is not None:
                value = result[parameter.root[5:]]
                loss = result[valueKey]
                key = str(roundPrecision(value, 3))
                if key in mergedResults:
                    mergedResults[key].append(loss)
                else:
                    mergedResults[key] = [loss]

        pairs = sorted(mergedResults.items(), key=lambda v: float(v[0]))
        values = [float(v[0]) for v in pairs]

        losses = []
        if bucket_reduction == 'mean':
            losses = [numpy.mean(v[1]) for v in pairs]
        elif bucket_reduction == 'median':
            losses = [numpy.median(v[1]) for v in pairs]
        elif bucket_reduction == 'min':
            losses = [numpy.min(v[1]) for v in pairs]

        threshhold = numpy.percentile(losses, cutoff * 100)

        filteredValues = []
        filteredLosses = []
        for index in range(len(losses)):
            if losses[index] < threshhold:
                filteredValues.append((values[index]))
                filteredLosses.append((losses[index]))

        if len(filteredValues) < 2:
            filteredValues = values
            filteredLosses = losses

        if numBuckets is not None:
            buckets = self.computeBucketsForParameter(results, parameter, numBuckets)

            newLosses = []
            newValues = []
            for bucketIndex, bucket in enumerate(buckets):
                bucketLosses = []
                for valueIndex, value in enumerate(filteredValues):
                    if (bucketIndex == 0 and value <= bucket) or (value > buckets[bucketIndex - 1] and value <= bucket):
                        bucketLosses.append(filteredLosses[valueIndex])
                if len(bucketLosses) > 0:
                    if bucket_reduction == 'mean':
                        newLosses.append(numpy.mean(bucketLosses))
                        newValues.append(bucket)
                    elif bucket_reduction == 'min':
                        newLosses.append(numpy.min(bucketLosses))
                        newValues.append(bucket)
                    elif bucket_reduction == 'median':
                        newLosses.append(numpy.median(bucketLosses))
                        newValues.append(bucket)
            filteredValues = newValues
            filteredLosses = newLosses

        bottom = numpy.min(filteredValues)
        top = numpy.max(filteredValues) + 1e-5
        trendLineXCoords = numpy.arange(bottom, top, (top - bottom) / 100)

        # If there are at least two results, we can fit a linear trend line
        linearTrendValues = None
        linearTrendLine = None
        if len(filteredLosses) > 2:
            z = numpy.polyfit(filteredValues, filteredLosses, 1, full=True)[0]
            p = numpy.poly1d(z)
            linearTrendValues = p(filteredValues)
            linearTrendLine = p(trendLineXCoords)

        medianValue = numpy.median(filteredValues)
        medianLoss = numpy.median(filteredLosses)

        # If there are at least three results, we can attempt to fit an exponential trend line
        exponentialTrendValues = None
        exponentialTrendLine = None
        if medianValue != 0:
            try:
                start_b = math.log10(medianLoss) / (-medianValue)
                # start_b = 1e-0
                if len(filteredLosses) > 2:
                    # Compute an exponential trend line
                    def exponenial_func(x, a, b, c):
                        return a * numpy.exp(-b * x) + c

                    warnings.simplefilter('ignore', scipy.optimize.OptimizeWarning)
                    oldErrorSettings = numpy.seterr(all='ignore')
                    popt, pcov = scipy.optimize.curve_fit(exponenial_func, filteredValues, filteredLosses, p0=(1, start_b, 1),
                                                          bounds=([-numpy.inf, -abs(start_b * 1e1), -numpy.inf], [+numpy.inf, +abs(start_b * 1e1), +numpy.inf]))
                    exponentialTrendValues = exponenial_func(numpy.array(filteredValues).copy(), *popt)
                    exponentialTrendLine = exponenial_func(numpy.array(trendLineXCoords).copy(), *popt)
                    numpy.seterr(**oldErrorSettings)

            except RuntimeError as e:
                # Scipy gives a run-time error if it fails to find an exponential curve that fits the data
                pass

        newResults = []
        for index in range(len(filteredLosses)):
            data = {
                parameter.root[5:]: filteredValues[index],
                valueKey: filteredLosses[index],
                "linearTrend": linearTrendValues[index] if linearTrendValues is not None else filteredLosses[index],
                "exponentialTrend": exponentialTrendValues[index] if exponentialTrendValues is not None else filteredLosses[index]
            }

            newResults.append(data)

        newResults = sorted(newResults, key=lambda val: val[parameter.root[5:]])

        return newResults, list(zip(trendLineXCoords, linearTrendLine)) if linearTrendLine is not None else None, list(
            zip(trendLineXCoords, exponentialTrendLine)) if exponentialTrendLine is not None else None

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
            if result['loss'] is not None:
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

        covarianceModel = sklearn.covariance.LedoitWolf()
        covarianceModel.fit(numpy.array(vectors), numpy.array(outputs))

        correlationModel = sklearn.linear_model.LinearRegression()
        correlationModel.fit(numpy.array(vectors), numpy.array(outputs))
        coefficients = correlationModel.coef_


        covariances = covarianceModel.covariance_
        correlations = numpy.zeros_like(covariances)
        deviations = numpy.std(vectors, axis=0)

        for label1Index in range(len(labels)):
            for label2Index in range(len(labels)):
                if label1Index != label2Index:
                    if deviations[label1Index] * deviations[label2Index] == 0:
                        correlations[label1Index][label2Index] = 0
                    else:
                        correlations[label1Index][label2Index] = covariances[label1Index][label2Index] / (deviations[label1Index] * deviations[label2Index])

        correlationScaling = (20.0) / (numpy.max(correlations) - numpy.min(correlations))
        correlations *= correlationScaling

        coefficientScaling = (20.0) / (numpy.max(coefficients) - numpy.min(coefficients))
        for label1Index in range(len(labels)):
            correlations[label1Index][label1Index] = coefficients[label1Index] * coefficientScaling

        for label1Index in range(len(labels)):
            for label2Index in range(len(labels)):
                correlations[label1Index][label2Index] = roundPrecision(correlations[label1Index][label2Index], 3)

        return correlations, labels

    def computeBucketsForParameter(self, results, parameter, numBuckets=10):
        values = [result[parameter.root[5:]] for result in results]

        bottom = numpy.min(values)
        top = numpy.max(values)
        if (top - bottom) < 1e-5:
            top += 1e-5

        # Determine the number of possible discrete values for this parameter (in the entire hyperparameter space).
        # If there are less possible discrete values then number of buckets, just return each of them as our buckets.
        if parameter.config.get('rounding', None) is not None:
            paramMin = parameter.config['min']
            paramMax = parameter.config['max']
            possibleValues = math.ceil((paramMax - paramMin) / parameter.config['rounding']) + 1
            if possibleValues <= numBuckets:
                discrete = list(numpy.arange(paramMin, paramMax, parameter.config['rounding'])) + [paramMax]

                # Add in one additional bucket on the low end.
                buckets = sorted(list(discrete))
                diff = buckets[1] - buckets[0]
                bottom = bottom - diff
                buckets.insert(0, bottom)
                return buckets

        # Count the number of discrete values for this parameter in our results.
        # If there are less discrete values then the number of buckets,
        # then modify the bottom so we consider the lowest discrete value
        # as its own bucket
        discrete = set(values)
        if len(discrete) <= numBuckets:
            # Add in one additional bucket on the low end.
            buckets = sorted(list(discrete))
            if len(buckets) > 1:
                diff = buckets[1] - buckets[0]
                bottom = bottom - diff
            else:
                bottom = buckets[0] - 1.0

        buckets = []
        if parameter.config.get('scaling', 'linear') == 'linear':
            domain = top - bottom
            buckets = [bottom]
            while len(buckets) <= (numBuckets):
                buckets = buckets + [buckets[-1] + (domain / numBuckets)]
        elif parameter.config.get('scaling', 'linear') == 'logarithmic':
            logMax = math.log(top)
            if bottom > 0:
                logMin = math.log(bottom)
            else:
                logMin = math.log(1e-7)
            domain = logMax - logMin

            logBuckets = [logMin]
            while len(logBuckets) <= numBuckets:
                logBuckets = logBuckets + [logBuckets[-1] + (domain / numBuckets)]
            buckets = [math.exp(n) for n in logBuckets]

        buckets = [value for value in buckets]

        return buckets

    def computeLossMatrix(self, results, parameter1, parameter2, valueKey='loss', cutoff=1.0, reduction='min'):
        """
            This computes the loss matrix between two hyper-parameters. The loss matrix
            helps you to visualize what are the best areas of the hyper parameter space
            by plotting them on a grid and coloring them.
        """
        losses = [numpy.min(v[valueKey]) for v in results if v[valueKey] is not None]
        threshhold = numpy.percentile(losses, cutoff * 100)
        filteredResults = [result for result in results if result[valueKey] is not None and result[valueKey] < threshhold]

        # Divide the range up into 100 parts
        numBuckets = 10
        parameter1Buckets = self.computeBucketsForParameter(filteredResults, parameter1, numBuckets)
        parameter2Buckets = self.computeBucketsForParameter(filteredResults, parameter2, numBuckets)

        # Create a grid for each of the values
        resultGrid = []
        for value in parameter1Buckets[1:]:
            row = []
            for value in parameter2Buckets[1:]:
                row.append([])
            resultGrid.append(row)

        # Go through each of the results, and put them into one of the buckets (or exclude them if these hyper-parameters were not active
        for result in results:
            if result[valueKey] is None:
                continue
            parameter1Key = parameter1.root[5:]
            parameter2Key = parameter2.root[5:]

            if isinstance(result[parameter1Key], float) or isinstance(result[parameter1Key], int) or isinstance(result[parameter1Key], bool):
                if isinstance(result[parameter2Key], float) or isinstance(result[parameter2Key], int) or isinstance(result[parameter2Key], bool):
                    parameter1Value = float(result[parameter1Key])
                    parameter2Value = float(result[parameter2Key])

                    parameter1Index = None
                    for index1, value1 in enumerate(parameter1Buckets[1:]):
                        if parameter1Value <= value1:
                            parameter1Index = index1
                            break
                    parameter2Index = None
                    for index2, value2 in enumerate(parameter2Buckets[1:]):
                        if parameter2Value <= value2:
                            parameter2Index = index2
                            break
                    if parameter1Index is not None and parameter2Index is not None:
                        resultGrid[parameter1Index][parameter2Index].append(result)

        # Now go through each entry in the grid and compute the average score
        scoreGrid = []
        for row in resultGrid:
            scoreRow = []
            for column in row:
                if len(column) > 0:
                    if reduction == 'min':
                        scoreRow.append(numpy.min([result[valueKey] for result in column if result[valueKey] is not None]))
                    elif reduction == 'mean':
                        scoreRow.append(numpy.mean([result[valueKey] for result in column if result[valueKey] is not None]))
                    elif reduction == 'median':
                        scoreRow.append(numpy.median([result[valueKey] for result in column if result[valueKey] is not None]))
                else:
                    scoreRow.append(None)
            scoreGrid.append(scoreRow)

        newScoreGrid = []
        for value in parameter1Buckets[1:]:
            row = []
            for value in parameter2Buckets[1:]:
                row.append(0)
            newScoreGrid.append(row)

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
                    if len(closestValues) > 0:
                        # Take the mean of the closest values and use that for this location on the score grid
                        newScoreGrid[rowIndex][columnIndex] = numpy.mean(closestValues)
                    else:
                        newScoreGrid[rowIndex][columnIndex] = 0
                else:
                    newScoreGrid[rowIndex][columnIndex] = scoreGrid[rowIndex][columnIndex]

        # Round all of the values in the final score grid. This makes them more pleasant to look at for display purposes.
        for rowIndex, row in enumerate(newScoreGrid):
            for columnIndex, column in enumerate(row):
                newScoreGrid[rowIndex][columnIndex] = roundPrecision(newScoreGrid[rowIndex][columnIndex])

        # Transpose the score grid, so that parameter1 is on the horizontal
        newScoreGrid = numpy.transpose(numpy.array(newScoreGrid))
        newScoreGrid = numpy.flip(newScoreGrid, axis=0)

        parameter2Buckets.reverse()

        return newScoreGrid, parameter1Buckets, parameter2Buckets
