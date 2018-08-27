import os.path
import json
import traceback
import csv
import copy
import sklearn.preprocessing
import sklearn.neighbors
import random
import pickle
import lightgbm
import numpy
from pprint import pprint
from simulation import AlgorithmSimulation

dirs = os.listdir('.')

atpeParameterKeys = [
    'gamma',
    'nEICandidates',
    'resultFilteringAgeMultiplier',
    'resultFilteringLossRankMultiplier',
    'resultFilteringMode',
    'resultFilteringRandomProbability',
    'secondaryCorrelationExponent',
    'secondaryCorrelationMultiplier',
    'secondaryCutoff',
    # 'secondarySorting',
    'secondaryFixedProbability',
    'secondaryLockingMode',
    'secondaryProbabilityMode',
    'secondaryTopLockingPercentile',
]

# We cascade the predictions - feeding prior predictions into the next one. There is a specific ordering to it, in order to help regularize the model based on the way our results were sampled
predictorKeyCascadeOrdering = [
    'resultFilteringMode',
    'secondaryLockingMode',
    'secondaryProbabilityMode',
    'resultFilteringAgeMultiplier',
    'resultFilteringLossRankMultiplier',
    'resultFilteringRandomProbability',
    'secondaryTopLockingPercentile',
    'secondaryCorrelationExponent',
    'secondaryCorrelationMultiplier',
    'secondaryFixedProbability',
    'secondaryCutoff',
    'gamma',
    'nEICandidates'
]


atpeParameterPredictionStandardDeviationRatio = {
    'gamma': 0.9,
    'nEICandidates': 0.5,
    'resultFilteringAgeMultiplier': 1.0,
    'resultFilteringLossRankMultiplier': 1.0,
    'resultFilteringRandomProbability': 1.0,
    'secondaryCorrelationExponent': 1.0,
    'secondaryCorrelationMultiplier': 1.0,
    'secondaryCutoff': 0.9,
    'secondaryFixedProbability': 1.0,
    'secondaryTopLockingPercentile': 1.0
}

# Custom params on per atpe param basis for training the lightgbm models
customLightGBMParams = {
    'secondaryCutoff': {
        'feature_fraction': 0.7 # Extra bagging required on these ones for good generalization, since they are late in the cycle and can fit to other atpe parameter predictions
    },
    'gamma': {
        'feature_fraction': 0.7 # Extra bagging required on these ones for good generalization, since they are late in the cycle and can fit to other atpe parameter predictions
    }
}

classPredictorKeys = [
    'resultFilteringMode',
    'secondaryLockingMode',
    'secondaryProbabilityMode'
]

numPredictorClasses = {
    'resultFilteringMode': 4,
    'secondaryLockingMode': 2,
    'secondaryProbabilityMode': 2
}

atpeParameterValues = {
    'resultFilteringMode': ['age', 'loss_rank', 'none', 'random'],
    'secondaryLockingMode': ['random', 'top'],
    'secondaryProbabilityMode': ['correlation', 'fixed']
}

nonFeatureKeys = [
    'algorithm',
    'contributions_exponential',
    'contributions_linear',
    'contributions_logarithmic',
    'contributions_peakvalley',
    'contributions_random',
    'fail_rate',
    'history',
    'interactions',
    'interactions_index',
    'interactions_linear',
    'interactions_peakvalley',
    'interactions_random',
    'interactions_wave',
    'loss',
    'noise',
    'run',
    'time',
    'trial',
    'log10_trial',
    'num_parameters'
]

featureKeys = [
    'all_correlation_best_percentile25_ratio',
    'all_correlation_best_percentile50_ratio',
    'all_correlation_best_percentile75_ratio',
    'all_correlation_kurtosis',
    'all_correlation_percentile5_percentile25_ratio',
    'all_correlation_skew',
    'all_correlation_stddev_best_ratio',
    'all_correlation_stddev_median_ratio',
    'all_loss_best_percentile25_ratio',
    'all_loss_best_percentile50_ratio',
    'all_loss_best_percentile75_ratio',
    'all_loss_kurtosis',
    'all_loss_percentile5_percentile25_ratio',
    'all_loss_skew',
    'all_loss_stddev_best_ratio',
    'all_loss_stddev_median_ratio',
    'log10_cardinality',
    'recent_10_correlation_best_percentile25_ratio',
    'recent_10_correlation_best_percentile50_ratio',
    'recent_10_correlation_best_percentile75_ratio',
    'recent_10_correlation_kurtosis',
    'recent_10_correlation_percentile5_percentile25_ratio',
    'recent_10_correlation_skew',
    'recent_10_correlation_stddev_best_ratio',
    'recent_10_correlation_stddev_median_ratio',
    'recent_10_loss_best_percentile25_ratio',
    'recent_10_loss_best_percentile50_ratio',
    'recent_10_loss_best_percentile75_ratio',
    'recent_10_loss_kurtosis',
    'recent_10_loss_percentile5_percentile25_ratio',
    'recent_10_loss_skew',
    'recent_10_loss_stddev_best_ratio',
    'recent_10_loss_stddev_median_ratio',
    'recent_15%_correlation_best_percentile25_ratio',
    'recent_15%_correlation_best_percentile50_ratio',
    'recent_15%_correlation_best_percentile75_ratio',
    'recent_15%_correlation_kurtosis',
    'recent_15%_correlation_percentile5_percentile25_ratio',
    'recent_15%_correlation_skew',
    'recent_15%_correlation_stddev_best_ratio',
    'recent_15%_correlation_stddev_median_ratio',
    'recent_15%_loss_best_percentile25_ratio',
    'recent_15%_loss_best_percentile50_ratio',
    'recent_15%_loss_best_percentile75_ratio',
    'recent_15%_loss_kurtosis',
    'recent_15%_loss_percentile5_percentile25_ratio',
    'recent_15%_loss_skew',
    'recent_15%_loss_stddev_best_ratio',
    'recent_15%_loss_stddev_median_ratio',
    'recent_25_correlation_best_percentile25_ratio',
    'recent_25_correlation_best_percentile50_ratio',
    'recent_25_correlation_best_percentile75_ratio',
    'recent_25_correlation_kurtosis',
    'recent_25_correlation_percentile5_percentile25_ratio',
    'recent_25_correlation_skew',
    'recent_25_correlation_stddev_best_ratio',
    'recent_25_correlation_stddev_median_ratio',
    'recent_25_loss_best_percentile25_ratio',
    'recent_25_loss_best_percentile50_ratio',
    'recent_25_loss_best_percentile75_ratio',
    'recent_25_loss_kurtosis',
    'recent_25_loss_percentile5_percentile25_ratio',
    'recent_25_loss_skew',
    'recent_25_loss_stddev_best_ratio',
    'recent_25_loss_stddev_median_ratio',
    'top_10%_correlation_best_percentile25_ratio',
    'top_10%_correlation_best_percentile50_ratio',
    'top_10%_correlation_best_percentile75_ratio',
    'top_10%_correlation_kurtosis',
    'top_10%_correlation_percentile5_percentile25_ratio',
    'top_10%_correlation_skew',
    'top_10%_correlation_stddev_best_ratio',
    'top_10%_correlation_stddev_median_ratio',
    'top_10%_loss_best_percentile25_ratio',
    'top_10%_loss_best_percentile50_ratio',
    'top_10%_loss_best_percentile75_ratio',
    'top_10%_loss_kurtosis',
    'top_10%_loss_percentile5_percentile25_ratio',
    'top_10%_loss_skew',
    'top_10%_loss_stddev_best_ratio',
    'top_10%_loss_stddev_median_ratio',
    'top_20%_correlation_best_percentile25_ratio',
    'top_20%_correlation_best_percentile50_ratio',
    'top_20%_correlation_best_percentile75_ratio',
    'top_20%_correlation_kurtosis',
    'top_20%_correlation_percentile5_percentile25_ratio',
    'top_20%_correlation_skew',
    'top_20%_correlation_stddev_best_ratio',
    'top_20%_correlation_stddev_median_ratio',
    'top_20%_loss_best_percentile25_ratio',
    'top_20%_loss_best_percentile50_ratio',
    'top_20%_loss_best_percentile75_ratio',
    'top_20%_loss_kurtosis',
    'top_20%_loss_percentile5_percentile25_ratio',
    'top_20%_loss_skew',
    'top_20%_loss_stddev_best_ratio',
    'top_20%_loss_stddev_median_ratio',
    'top_30%_correlation_best_percentile25_ratio',
    'top_30%_correlation_best_percentile50_ratio',
    'top_30%_correlation_best_percentile75_ratio',
    'top_30%_correlation_kurtosis',
    'top_30%_correlation_percentile5_percentile25_ratio',
    'top_30%_correlation_skew',
    'top_30%_correlation_stddev_best_ratio',
    'top_30%_correlation_stddev_median_ratio',
    'top_30%_loss_best_percentile25_ratio',
    'top_30%_loss_best_percentile50_ratio',
    'top_30%_loss_best_percentile75_ratio',
    'top_30%_loss_kurtosis',
    'top_30%_loss_percentile5_percentile25_ratio',
    'top_30%_loss_skew',
    'top_30%_loss_stddev_best_ratio',
    'top_30%_loss_stddev_median_ratio'
]


def roundPrecision(number, precision=4):
    """ Rounds the given floating point number to a certain precision, for output."""
    return float(('{:.' + str(precision) + 'E}').format(number))


def preprocessResult(result):
    # Here we preprocess results as early as possible to create different predictor keys
    # result['secondarySorting'] = (1 if float(result['secondaryCutoff']) > 0 else (-1 if float(result['secondaryCutoff']) < 0 else 0))
    # result['secondaryCutoff'] = abs(float(result['secondaryCutoff']))
    return result


def extractResultsFromLogs():
    # This file merges together all the results
    dirs = sorted(os.listdir('.'))

    allFails = []
    allResults = []

    for dir in dirs:
        if 'run' not in dir:
            continue
        if os.path.exists(os.path.join(dir, 'hypermax', 'nohup.out')):
            with open(os.path.join(dir, 'hypermax', 'nohup.out'), 'rt') as file:
                text = file.read()

                fails = []
                results = []

                # Extract each of the results out of the log files
                start = text.find('{')
                while start != -1:
                    end = text.find('}', start)
                    result = text[start:end + 1]
                    result = result.replace('\'', '"')
                    result = result.replace('None', 'null')
                    try:
                        data = json.loads(result)
                        data['run'] = dir
                        results.append(preprocessResult(data))
                    except Exception:
                        fails.append(result)
                        # traceback.print_exc()
                    start = text.find('{', end)

                allResults = allResults + results
                allFails = allFails + fails
    return allResults


def extractResultsFromCSVs():
    # This file merges together all the results
    dirs = sorted(os.listdir('.'))

    allResults = []

    for dir in dirs:
        if 'run' not in dir:
            continue
        filePath = os.path.join(dir, 'hypermax', 'results.csv')
        if os.path.exists(filePath):
            with open(filePath, 'rt') as file:
                results = [dict(result) for result in csv.DictReader(file)]
                for result in results:
                    result['run'] = dir
                allResults = allResults + [preprocessResult(result) for result in results]
    return allResults


def getDeduplicatedResults():
    logResults = extractResultsFromLogs()
    csvResults = extractResultsFromCSVs()

    logResultsByRun = {}
    csvResultsByRun = {}
    for result in logResults:
        # Make sure its a valid result. A very small number of results are missing all the requisite features (for an unknown reason, perhaps because they were killed part way into output
        valid = True
        for key in featureKeys:
            if key not in result:
                valid=False
        if not valid:
            continue
        if result['run'] in logResultsByRun:
            logResultsByRun[result['run']].append(result)
        else:
            logResultsByRun[result['run']] = [result]
    for result in csvResults:
        if result['run'] in csvResultsByRun:
            csvResultsByRun[result['run']].append(result)
        else:
            csvResultsByRun[result['run']] = [result]

    duplicates = []
    additionals = []
    csvResultsByRunCloned = copy.deepcopy(csvResultsByRun)
    for run in logResultsByRun.keys():
        runDuplicates = []
        runAdditionals = []

        model = None
        if run in csvResultsByRun:
            # Build a nearest neighbor model to help us find the duplicates
            model = sklearn.neighbors.NearestNeighbors(n_neighbors=10)
            vectors = []
            for result in csvResultsByRunCloned[run]:
                vectors.append([result[key] for key in featureKeys])
            model.fit(vectors)

        def testCSVResult(result2):
            same = True
            for key in result.keys():
                if result[key] is not None and result2[key] is not None:
                    try:
                        same = not (abs(float(result[key]) - float(result2[key])) > 0.01)
                    except ValueError:
                        same = (result[key] == result2[key])

                    if not same:
                        break
            return same

        for result in logResultsByRun[run]:
            found = False
            if run in csvResultsByRunCloned:
                vector = [result[key] for key in featureKeys]
                nearest = model.kneighbors([vector], n_neighbors=10) # The k nearest neighbors is good enough to always find the duplicates.
                for neighbor in nearest[1][0]:
                    found = testCSVResult(csvResultsByRun[run][neighbor])
                    if found:
                        break
            if found:
                runDuplicates.append(result)
            if not found:
                runAdditionals.append(result)

        print(run, 'dupes', len(runDuplicates))
        print(run, 'adds', len(runAdditionals))
        duplicates = duplicates + runDuplicates
        additionals = additionals + runAdditionals

    allResults = csvResults + additionals
    return allResults


def mergeResults():
    allResults = getDeduplicatedResults()

    # with open('final/allResults.csv', 'rt') as file:
    #     results = [dict(result) for result in csv.DictReader(file)]
    #     for result in results:
    #         del result['blank1']
    #         del result['blank2']
    #     allResults = results

    # Convert everything to floats where possible
    for result in allResults:
        for key in result.keys():
            if result[key] is not None and result[key] != '':
                try:
                    result[key] = float(result[key])
                except ValueError:
                    pass

    # Filter out all results with a loss of 1.0. These results are meaningless for our dataset, since they are caused our random-failure noise, and not by actually searching the dataset
    allResults = [result for result in allResults if float(result['loss']) < 1.0]

    for result in allResults:
        if 'algorithm' not in result:
            result['algorithm'] = ''
        if 'time' not in result:
            result['time'] = ''

    if not os.path.exists('final'):
        os.mkdir('final')
    if not os.path.exists('final/algos'):
        os.mkdir('final/algos')

    savedAlgorithms = {}

    for result in allResults:
        result['blank1'] = ''
        result['blank2'] = ''

        if result['algorithm'] and (result['run'] + result['algorithm']) not in savedAlgorithms:
            algoFileName = os.path.join(result['run'], 'hypermax', result['algorithm'])
            if os.path.exists(algoFileName):
                data = pickle.load(open(algoFileName, 'rb'))['algo']

                scriptName = "algorithm-" + str(len(savedAlgorithms)) + ".py"
                dataName = "algorithm-" + str(len(savedAlgorithms)) + "-pickle.bin"
                with open(os.path.join('final', 'algos', scriptName), 'wt') as file:
                    file.write(data.computeScript)
                with open(os.path.join('final', 'algos', dataName), 'wb') as file:
                    pickle.dump(data, file)
                savedAlgorithms[result['run'] + result['algorithm']] = scriptName
                result['algorithm'] = scriptName
            else:
                result['algorithm'] = ''
        elif result['algorithm']:
            result['algorithm'] = savedAlgorithms[result['run'] + result['algorithm']]

    prettyResults = []
    for result in allResults:
        prettyResult = {}
        for key in result.keys():
            prettyResult[key] = result[key]
            if prettyResult[key]:
                try:
                    number = roundPrecision(float(prettyResult[key]))
                    prettyResult[key] = number
                except ValueError:
                    pass
                except TypeError:
                    pass
        prettyResults.append(prettyResult)


    with open("final/allResults.csv", "wt") as file:
        writer = csv.DictWriter(file, fieldnames=(atpeParameterKeys + ['blank1'] + featureKeys + ['blank2'] + nonFeatureKeys))
        writer.writeheader()
        writer.writerows(prettyResults)

    # Compute the normalization region for each predictor
    scalers = {}
    for feature in featureKeys:
        vectors = []
        for result in allResults:
            if float(result[feature]) != 0:
                vectors.append([result[feature]])

        vectors = numpy.array(vectors)
        # Use percentiles to focus our scaler on the most common values, making it more immune to the weird outliers in our dataset
        percentile20 = numpy.percentile(vectors[:,0], q=20)
        percentile80 = numpy.percentile(vectors[:,0], q=80)
        vectors = [vector for vector in vectors if vector[0] > percentile20 and vector[0] < percentile80]

        scaler = sklearn.preprocessing.StandardScaler(vectors)
        scaler.fit(vectors)

        scalers[feature] = {
            'scales': scaler.scale_.tolist(),
            'means': scaler.mean_.tolist(),
            'variances': scaler.var_.tolist()
        }
    with open("scaling_model.json", 'wt') as file:
        json.dump(scalers, file)

    for keyIndex, key in enumerate(featureKeys):
        featureScalingModel = sklearn.preprocessing.StandardScaler()
        featureScalingModel.scale_ = numpy.array(scalers[key]['scales'])
        featureScalingModel.mean_ = numpy.array(scalers[key]['means'])
        featureScalingModel.var_ = numpy.array(scalers[key]['variances'])

        vectors = []
        for resultIndex, result in enumerate(allResults):
            vectors.append([float(result[key])])

        vectors = featureScalingModel.transform(vectors)
        for resultIndex, result in enumerate(allResults):
            result[key] = roundPrecision(vectors[resultIndex][0])


    # Output the normalized results
    prettyResults = []
    for result in allResults:
        prettyResult = {}
        for key in result.keys():
            prettyResult[key] = result[key]
            if prettyResult[key]:
                try:
                    number = roundPrecision(float(prettyResult[key]))
                    prettyResult[key] = number
                except ValueError:
                    pass
                except TypeError:
                    pass
        prettyResults.append(prettyResult)

    with open("final/allResults_normalized.csv", "wt") as file:
        writer = csv.DictWriter(file, fieldnames=atpeParameterKeys + ['blank1'] + featureKeys + ['blank2'] + nonFeatureKeys)
        writer.writeheader()
        writer.writerows(prettyResults)

    # Now put together the dataset for each predictor feature, so its more convenient to build models on them
    shuffled = list(allResults)
    random.shuffle(shuffled)
    cutoff = int(len(shuffled) * 0.2)
    testing = shuffled[:cutoff]
    training = shuffled[cutoff:]

    def writeDataset(key, filename, dataset):
        with open(filename, 'wt') as file:
            keyResults = []
            for result in dataset:
                if result[key]:
                    data = {}
                    data[key] = result[key]
                    for feature in featureKeys:
                        data[feature] = result[feature]
                    keyResults.append(data)

            writer = csv.DictWriter(file, fieldnames=[key] + featureKeys)
            writer.writeheader()
            writer.writerows(keyResults)

    for key in atpeParameterKeys:
        writeDataset(key, "final/" + key + "_testing.csv", testing)
        writeDataset(key, "final/" + key + "_training.csv", training)

    return training, testing



def trainATPEModels():
    with open('final/allResults_normalized.csv', 'rt') as file:
        allResults = [dict(result) for result in csv.DictReader(file)]

    # Now put together the dataset for each predictor feature, so its more convenient to build models on them
    shuffled = list(allResults)
    random.shuffle(shuffled)
    cutoff = int(len(shuffled) * 0.2)
    testing = shuffled[:cutoff]
    training = shuffled[cutoff:]


    def createDataset(key, dataset, atpeParamFeatures):
        vectors = []
        targets = []
        allTargets = set()
        for result in dataset:
            if result[key] not in allTargets:
                allTargets.add(result[key])

        allTargets = sorted(list(allTargets))

        names = copy.copy(featureKeys)
        for atpeParamFeature in atpeParamFeatures:
            if atpeParamFeature in atpeParameterValues:
                for value in atpeParameterValues[atpeParamFeature]:
                    names.append(atpeParamFeature + "_" + value)
            else:
                names.append(atpeParamFeature)

        for result in dataset:
            if result[key]:
                vector = []
                for feature in featureKeys:
                    vector.append(float(result[feature]))
                for atpeParamFeature in atpeParamFeatures:
                    if atpeParamFeature in result and result[atpeParamFeature] is not None and result[atpeParamFeature] != '':
                        if atpeParamFeature in atpeParameterValues:
                            for value in atpeParameterValues[atpeParamFeature]:
                                vector.append(1.0 if result[atpeParamFeature] == value else 0)
                        else:
                            vector.append(float(result[atpeParamFeature]))
                    else:
                        vector.append(-3) # We use -3 because none of our atpe parameters ever take this value
                vectors.append(vector)

                if key in classPredictorKeys:
                    targets.append(allTargets.index(result[key]))
                else:
                    targets.append(float(result[key]))
        return lightgbm.Dataset(numpy.array(vectors), label=numpy.array(targets), feature_name=names)


    allModels = []
    for keyIndex, key in enumerate(predictorKeyCascadeOrdering):
        atpeParamFeatures = predictorKeyCascadeOrdering[:keyIndex]
        trainingData = createDataset(key, training, atpeParamFeatures=atpeParamFeatures)
        testingData = createDataset(key, testing, atpeParamFeatures=atpeParamFeatures)

        allFeatureNames = trainingData.feature_name

        params = {
            'num_iterations': 250,
            'is_provide_training_metric': True,
            "early_stopping_round": 5,
            "feature_fraction": 0.90,
            "learning_rate": 0.05
        }

        if key in customLightGBMParams:
            for param in customLightGBMParams[key]:
                params[param] = customLightGBMParams[key][param]

        if key in classPredictorKeys:
            params['num_class'] = numPredictorClasses[key]
            params['objective'] = 'multiclass'
            params['metric'] = 'multi_error'
        else:
            params['objective'] = 'regression_l2'
            params['metric'] = 'l2'

        model = lightgbm.train(params, trainingData, valid_sets=[testingData], verbose_eval=False)

        model.save_model("model-" + key + ".txt")

        if key not in classPredictorKeys:
            # Now we determine the "adjustment factor". Because these models are trained on an extremely noisy data set,
            # We have to eliminate the central tendency that results from training on it, so that the outputs of our model
            # Take up the full range of possible ATPE parameter values
            print(key)
            origStddev = numpy.std([float(result[key]) for result in training if result[key] is not None and result[key] != ''])
            origMean = numpy.mean([float(result[key]) for result in training if result[key] is not None and result[key] != ''])

            trainingPredicted = []
            for result in training:
                vector = []
                for feature in featureKeys:
                    vector.append(result[feature])
                trainingPredicted.append(float(model.predict([vector])[0]))

            predStddev = numpy.std(trainingPredicted) / atpeParameterPredictionStandardDeviationRatio[key]
            predMean = numpy.mean(trainingPredicted)

            with open('model-' + key + "-configuration.json", 'wt') as file:
                json.dump({
                    "origStddev": origStddev,
                    "origMean": origMean,
                    "predStddev": predStddev,
                    "predMean": predMean
                }, file)

            def renormalize(value):
                return (((value - predMean) / predStddev) * origStddev) + origMean

            totalL1Error = 0
            totalL1NormalizedError = 0
            totalCount = 0
            with open('predictions-' + key + '.csv', 'wt') as file:
                writer = csv.DictWriter(file, fieldnames=[key, key + "_predicted", key + "_predicted_normalized", key + "_l1_error", key + "_l1_normalized_error"])
                writer.writeheader()
                for result in testing:
                    if result[key] is not None and result[key] != '':
                        vector = []
                        for feature in featureKeys:
                            vector.append(float(result[feature]))
                        for atpeParamFeature in atpeParamFeatures:
                            if atpeParamFeature in result and result[atpeParamFeature] is not None and result[atpeParamFeature] != '':
                                if atpeParamFeature in atpeParameterValues:
                                    for value in atpeParameterValues[atpeParamFeature]:
                                        vector.append(1.0 if result[atpeParamFeature] == value else 0)
                                else:
                                    vector.append(float(result[atpeParamFeature]))
                            else:
                                vector.append(-3) # We use -3 because none of our atpe parameters ever take this value
                        value = roundPrecision(float(model.predict([vector])[0]))
                        predicted = roundPrecision(float(renormalize(value)))
                        l1_error = roundPrecision(float(abs(value - float(result[key]))))
                        l1_normalized_error = roundPrecision(float(abs(predicted - float(result[key]))))
                        totalL1Error += l1_error
                        totalL1NormalizedError += l1_normalized_error
                        totalCount += 1
                        writer.writerow({
                            key: result[key],
                            key + "_predicted": value,
                            key + "_predicted_normalized": predicted,
                            key + "_l1_error": l1_error,
                            key + "_l1_normalized_error": l1_normalized_error,
                        })
            print("Average L1 Error:", totalL1Error/totalCount)
            print("Average Normalized L1 Error:", totalL1NormalizedError/totalCount)
        else:
            totalCorrect = 0
            totalCount = 0
            with open('predictions-' + key + '.csv', 'wt') as file:
                writer = csv.DictWriter(file, fieldnames=[key, key + "_predicted", key + "_correct"])
                writer.writeheader()
                for result in testing:
                    vector = []
                    for feature in featureKeys:
                        vector.append(float(result[feature]))
                    for atpeParamFeature in atpeParamFeatures:
                        if atpeParamFeature in result and result[atpeParamFeature] is not None and result[atpeParamFeature] != '':
                            if atpeParamFeature in atpeParameterValues:
                                for value in atpeParameterValues[atpeParamFeature]:
                                    vector.append(1.0 if result[atpeParamFeature] == value else 0)
                            else:
                                vector.append(float(result[atpeParamFeature]))
                        else:
                            vector.append(-3) # We use -3 because none of our atpe parameters ever take this value

                    predicted = atpeParameterValues[key][int(numpy.argmax(model.predict([vector])[0]))]
                    correct = (predicted == result[key])

                    probabilities = model.predict([vector])[0]*100
                    # print(probabilities)

                    totalCount += 1
                    if correct:
                        totalCorrect += 1

                    writer.writerow({
                        key: result[key],
                        key + "_predicted": predicted,
                        key + "_correct": correct
                    })
            print("Accuracy:", str(totalCorrect * 100 / totalCount), "%")


        importances = zip(allFeatureNames, model.feature_importance())
        importances = sorted(importances, key=lambda r:-r[1])
        print(key)
        for importance in importances:
            print("    ", importance[0], importance[1])

mergeResults()
trainATPEModels()