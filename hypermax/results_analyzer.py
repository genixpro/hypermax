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
import concurrent.futures
import colors
import traceback
import warnings
import scipy.optimize
import random



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
        self.chartNumber = 0

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
                    partialDirectory = os.path.join(*parts[:index + 1])
                else:
                    partialDirectory = parts[0]

                if not os.path.exists(partialDirectory):
                    os.mkdir(partialDirectory)

    def generateMultiParameterExports(self, results, parameter1, parameter2):
        try:
            subDirectory = os.path.join(self.directory, parameter1.root[5:], parameter2.root[5:])
            self.makeDirs(subDirectory)

            for reduction in ['mean', 'min']:
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

    def generateSingleParameterExports(self, results, parameter):
        try:
            subDirectory = os.path.join(self.directory, parameter.root[5:])
            self.makeDirs(subDirectory)

            lossCsvFilename = 'losses_' + parameter.root[5:] + '.csv'
            lossBucketedCsvFilename = 'losses_bucketed_' + parameter.root[5:] + '.csv'
            lossImageFilename = 'loss_chart_' + parameter.root[5:] + '.png'
            lossBucketedImageFilename = 'loss_chart_bucketed_' + parameter.root[5:] + '.png'
            lossTop10ImageFilename = 'loss_chart_top_10_percent_' + parameter.root[5:] + '.png'
            lossTop10BucketedImageFilename = 'loss_chart_top_10_percent_bucketed_' + parameter.root[5:] + '.png'
            self.exportSingleParameterLossCSV(os.path.join(subDirectory, lossCsvFilename), results, parameter, 'loss')
            self.exportSingleParameterLossCSV(os.path.join(subDirectory, lossBucketedCsvFilename), results, parameter, 'loss', numBuckets=20)
            self.exportSingleParameterLossChart(os.path.join(subDirectory, lossImageFilename), results, parameter, 'loss', 'Loss Chart')
            self.exportSingleParameterLossChart(os.path.join(subDirectory, lossBucketedImageFilename), results, parameter, 'loss', 'Loss Chart (20 Buckets)', numBuckets=20)
            self.exportSingleParameterLossChart(os.path.join(subDirectory, lossTop10ImageFilename), results, parameter, 'loss', 'Loss Chart (top 10%)', cutoff=0.1)
            self.exportSingleParameterLossChart(os.path.join(subDirectory, lossTop10BucketedImageFilename), results, parameter, 'loss', 'Loss Chart (20 buckets, top 10%)',
                                                cutoff=0.1, numBuckets=20)

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

    def outputResultsFolder(self, optimizer, detailed=True):
        # Ensure the directory we want to store results in is there
        self.makeDirs(self.directory)

        resultsFile = os.path.join(self.directory, 'results.csv')
        optimizer.exportResultsCSV(resultsFile)

        if len(optimizer.results) > 5:
            correlationsFile = os.path.join(self.directory, 'correlations.csv')
            self.exportCorrelationsToCSV(correlationsFile, optimizer)

        # Only do these results if detailed is enabled, since they take a lot more computation
        if len(optimizer.results) > 5 and detailed:
            parameters = Hyperparameter(optimizer.config.data['hyperparameters']).getFlatParameters()

            # We use concurrent.futures.ProcessThreadPool here for two reasons. One for speed (since generating the images can be slow)
            # The other is because  matplotlib is not inherently thread safe.
            futures = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                for parameter1 in parameters:
                    # self.generateSingleParameterExports(list(optimizer.results), parameter1)
                    futures.append(executor.submit(self.generateSingleParameterExports, list(optimizer.results), parameter1))
                    if parameter1.config['type'] == 'number':
                        for parameter2 in parameters:
                            if parameter2.config['type'] == 'number':
                                if parameter1.root != parameter2.root:
                                    # self.generateMultiParameterExports(list(optimizer.results), parameter1, parameter2)
                                    futures.append(executor.submit(self.generateMultiParameterExports, list(optimizer.results), parameter1, parameter2))
            for future in futures:
                if future.result() is not None:
                    print(traceback.format_exception_only(Exception, future.result()))
                    raise future.result()

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

    def exportTwoParameterScatter(self, fileName, results, parameter1, parameter2, valueKey='loss', cutoff=1.0, title="Scatter"):
        chartNumber = self.chartNumber
        self.chartNumber += 1

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

        minVal = float(numpy.min(losses))
        maxVal = float(numpy.max(losses))
        redVal = float(numpy.percentile(losses, q=90))
        yellowVal = float(numpy.percentile(losses, q=40))
        greenVal = float(numpy.percentile(losses, q=20))
        blueVal = float(numpy.percentile(losses, q=5))

        redVal = (redVal - minVal) / (maxVal - minVal)
        yellowVal = (yellowVal - minVal) / (maxVal - minVal)
        greenVal = (greenVal - minVal) / (maxVal - minVal)
        blueVal = (blueVal - minVal) / (maxVal - minVal)

        green = numpy.array([0, 1, 0, 1])
        yellow = numpy.array([1, 1, 0, 1])
        red = numpy.array([1, 0, 0, 1])
        blue = numpy.array([0.3, 0.3, 1, 1])

        fig, ax = plt.subplots()

        # We shuffle the results here because the scatter plot will by default draw later points over earlier ones, and later results will be on average better
        # anyhow, skewing the graphic
        resultsShuffled = list(results)
        random.shuffle(resultsShuffled)
        for key, resultScores in resultsMerged.items():
            xCoords.append(key[0])
            yCoords.append(key[1])
            scores.append(float((numpy.min(resultScores) - minVal) / (maxVal - minVal)))

        colorList = [(0, blue), (blueVal, blue), (greenVal, green), (yellowVal, yellow), (redVal, red), (1.0, red)]
        plt.scatter(xCoords, yCoords, c=numpy.array(scores), cmap=matplotlib.colors.LinearSegmentedColormap.from_list('loss_matrix_' + str(chartNumber), colorList, N=10240))

        plt.title(title + " of " + parameter1.root[5:] + " vs " + parameter2.root[5:], fontdict={"fontsize": 10})

        ax.set_xlabel(parameter1Key)
        ax.set_ylabel(parameter2Key)

        plt.tight_layout()
        plt.savefig(fileName, dpi=200)
        plt.close()

    def exportLossMatrixToCSV(self, fileName, results, parameter1, parameter2, valueKey='loss', cutoff=1.0, reduction='min'):
        scores, parameter1Buckets, parameter2Buckets = self.computeLossMatrix(results, parameter1, parameter2, valueKey, cutoff=cutoff, reduction=reduction)

        with open(fileName, 'wt') as file:
            writer = csv.writer(file)
            param1Padding = int((len(parameter1Buckets) - 1) / 2) - 1
            param2Padding = int((len(parameter2Buckets) - 1) / 2)

            writer.writerow([''] + ([''] * param2Padding) + [parameter2.root[5:]] + ([''] * (param2Padding)))
            writer.writerow(['', ''] + parameter2Buckets)
            for rowIndex, row in enumerate(scores):
                if rowIndex == param1Padding:
                    writer.writerow([parameter1.root[5:], str(parameter1Buckets[rowIndex])] + row)
                else:
                    writer.writerow(['', str(parameter1Buckets[rowIndex])] + row)

    def exportLossMatrixToImage(self, fileName, results, parameter1, parameter2, valueKey='loss', title='Loss Matrix', cutoff=1.0, mode='global', reduction='min'):
        chartNumber = self.chartNumber
        self.chartNumber += 1

        scores, parameter1Buckets, parameter2Buckets = self.computeLossMatrix(results, parameter1, parameter2, valueKey, cutoff=cutoff, reduction=reduction)

        minVal = float(numpy.min(scores))
        maxVal = float(numpy.max(scores))
        redVal = float(numpy.percentile(scores, q=90))
        yellowVal = float(numpy.percentile(scores, q=40))
        greenVal = float(numpy.percentile(scores, q=20))
        blueVal = float(numpy.percentile(scores, q=5))

        if mode == 'global':
            redVal = (redVal - minVal) / (maxVal - minVal)
            yellowVal = (yellowVal - minVal) / (maxVal - minVal)
            greenVal = (greenVal - minVal) / (maxVal - minVal)
            blueVal = (blueVal - minVal) / (maxVal - minVal)
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

        fig, ax = plt.subplots()

        colorList = [(0, blue), (blueVal, blue), (greenVal, green), (yellowVal, yellow), (redVal, red), (1.0, red)]

        im = ax.imshow(numpy.array(colorGrid, dtype=numpy.float32), cmap=matplotlib.colors.LinearSegmentedColormap.from_list('loss_matrix' + str(chartNumber), colorList, N=10240), interpolation='bicubic')

        # We want to show all ticks...
        ax.set_xticks(numpy.arange(len(parameter2Buckets)))
        ax.set_yticks(numpy.arange(len(parameter1Buckets)))

        # ... and label them with the respective list entries
        ax.set_xticklabels(parameter2Buckets)
        ax.set_yticklabels(parameter1Buckets)

        ax.set_xlabel(parameter1.root[5:])
        ax.set_ylabel(parameter2.root[5:])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Function which formats the text for display in each cell of heatmap
        def getText(i, j):
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
        fontSize = int(12 - max(0, 1.25 * (longest - 3)))
        for i in range(len(parameter1Buckets)):
            for j in range(len(parameter2Buckets)):
                ax.text(j, i, getText(i, j), ha="center", va="center", color="black", fontsize=fontSize)

        ax.set_title(title + " of " + parameter1.root[5:] + " vs " + parameter2.root[5:], fontdict={"fontsize": 10})
        plt.tight_layout()
        plt.savefig(fileName, dpi=200)
        plt.close()

    def exportSingleParameterLossChart(self, fileName, results, parameter, valueKey='loss', title='Loss Chart', cutoff=1.0, numBuckets=None):
        chartNumber = self.chartNumber
        self.chartNumber += 1

        values, linearTrendLine, exponentialTrendLine = self.computeParameterResultValues(results, parameter, valueKey, cutoff, numBuckets)

        fig, ax = plt.subplots()

        plt.title(title + " for " + parameter.root[5:])

        if parameter.config.get('scaling', 'linear') == 'logarithmic':
            plt.xscale('log')
        else:
            plt.xscale('linear')

        xCoords = [value[parameter.root[5:]] for value in values]
        yCoords = [value[valueKey] for value in values]

        ax.set_xlabel(parameter.root[5:])
        ax.set_ylabel(valueKey)

        plt.scatter(xCoords, yCoords)

        # Preserve the limits of the scatter graph when we apply the trend line
        xlim = plt.xlim()
        ylim = plt.ylim()

        if linearTrendLine and exponentialTrendLine:
            trendLineXCoords = [linearTrendLine[index][0] for index in range(len(linearTrendLine))]
            plt.plot(trendLineXCoords, [(linearTrendLine[index][1], exponentialTrendLine[index][1]) for index in range(len(exponentialTrendLine))], color='red', linestyle='dashed')
        elif linearTrendLine:
            trendLineXCoords = [linearTrendLine[index][0] for index in range(len(linearTrendLine))]
            plt.plot(trendLineXCoords, [linearTrendLine[index][1] for index in range(len(exponentialTrendLine))], color='red', linestyle='dashed')
        elif exponentialTrendLine:
            trendLineXCoords = [exponentialTrendLine[index][0] for index in range(len(linearTrendLine))]
            plt.plot(trendLineXCoords, [exponentialTrendLine[index][1] for index in range(len(exponentialTrendLine))], color='red', linestyle='dashed')

        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.tight_layout()
        plt.savefig(fileName, dpi=200)
        plt.close()

    def exportSingleParameterLossCSV(self, fileName, results, parameter, valueKey='loss', numBuckets=None):
        newResults, linearTrendLine, exponentialTrendLine = self.computeParameterResultValues(results, parameter, valueKey, cutoff=1.0, numBuckets=numBuckets)

        with open(fileName, 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=[parameter.root[5:], valueKey, 'linearTrend', 'exponentialTrend'], dialect='unix')
            writer.writeheader()
            writer.writerows(newResults)

    def computeParameterResultValues(self, results, parameter, valueKey='loss', cutoff=1.0, numBuckets=None):
        mergedResults = {}
        for result in results:
            if isinstance(result[parameter.root[5:]], float) and result[valueKey] is not None:
                value = result[parameter.root[5:]]
                loss = result[valueKey]
                key = str(roundPrecision(value, 3))
                if key in mergedResults:
                    mergedResults[key].append(loss)
                else:
                    mergedResults[key] = [loss]

        pairs = sorted(mergedResults.items(), key=lambda v: v[0])
        values = [float(v[0]) for v in pairs]
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
                    if (bucketIndex == 0 and value < bucket) or (value >= buckets[bucketIndex - 1] and value < bucket):
                        bucketLosses.append(filteredLosses[valueIndex])
                if len(bucketLosses) > 0:
                    newLosses.append(numpy.mean(bucketLosses))
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
                    popt, pcov = scipy.optimize.curve_fit(exponenial_func, filteredValues, filteredLosses, p0=(1, start_b, 1),
                                                          bounds=([-numpy.inf, -abs(start_b * 1e1), -numpy.inf], [+numpy.inf, +abs(start_b * 1e1), +numpy.inf]))
                    exponentialTrendValues = exponenial_func(numpy.array(filteredValues).copy(), *popt)
                    exponentialTrendLine = exponenial_func(numpy.array(trendLineXCoords).copy(), *popt)

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

        return newResults, list(zip(trendLineXCoords, linearTrendLine)) if exponentialTrendLine is not None else None, list(
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

    def computeBucketsForParameter(self, results, parameter, numBuckets=10):
        values = [result[parameter.root[5:]] for result in results]

        bottom = numpy.min(values)
        top = numpy.max(values)
        if (top - bottom) < 1e-5:
            top += 1e-5

        buckets = []
        if parameter.config['scaling'] == 'linear':
            domain = top - bottom
            buckets = list(numpy.arange(bottom, top + (domain / numBuckets), domain / numBuckets))[1:]
        elif parameter.config['scaling'] == 'logarithmic':
            logMax = math.log(top)
            logMin = math.log(bottom)
            domain = logMax - logMin
            logBuckets = numpy.arange(logMin, logMax + (domain / numBuckets), domain / numBuckets)[1:]
            buckets = [math.exp(n) for n in logBuckets]

        # Round the precision of the buckets down. Helps with the formatting of the charts
        buckets = [roundPrecision(value) for value in buckets]

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
        for value in parameter1Buckets:
            row = []
            for value in parameter2Buckets:
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
                    for index1, value1 in enumerate(parameter1Buckets):
                        if parameter1Value <= value1:
                            parameter1Index = index1
                            break
                    parameter2Index = None
                    for index2, value2 in enumerate(parameter2Buckets):
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
                else:
                    scoreRow.append(None)
            scoreGrid.append(scoreRow)

        newScoreGrid = []
        for value in parameter1Buckets:
            row = []
            for value in parameter2Buckets:
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

        return newScoreGrid, parameter1Buckets, parameter2Buckets
