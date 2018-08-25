import os.path
import json
import traceback
import csv
import copy
import sklearn.preprocessing
import random
import pickle
import lightgbm
import numpy
from pprint import pprint
from simulation import AlgorithmSimulation

dirs = os.listdir('.')

predictorKeys = [
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
    'trial'
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
    'log10_trial',
    'num_parameters',
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


def roundPrecision(number, precision=3):
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
                results = list(csv.DictReader(file))
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
        print(run, 'total', len(logResultsByRun[run]))
        for result in logResultsByRun[run]:
            found = False
            if run in csvResultsByRunCloned:
                for result2 in csvResultsByRunCloned[run]:
                    same = True
                    for key in result.keys():
                        if result[key] is not None and result2[key] is not None:
                            try:
                                same = not (abs(float(result[key]) - float(result2[key])) > 0.01)
                            except ValueError:
                                same = (result[key] == result2[key])

                            if not same:
                                break
                    if same:
                        found = True
                        runDuplicates.append(result)
                        csvResultsByRunCloned[run].remove(result2)
                        break
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
    # allResults = extractResultsFromCSVs()

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
                    number = roundPrecision(float(prettyResult[key]), precision=4)
                    prettyResult[key] = number
                except ValueError:
                    pass
                except TypeError:
                    pass
        prettyResults.append(prettyResult)


    with open("final/allResults.csv", "wt") as file:
        writer = csv.DictWriter(file, fieldnames=(predictorKeys + ['blank1'] + featureKeys + ['blank2'] + nonFeatureKeys))
        writer.writeheader()
        writer.writerows(prettyResults)

    # # Normalize each predictor
    # vectors = []
    # for result in allResults:
    #     vector = []
    #     for key in featureKeys:
    #         vector.append(result[key])
    #     vectors.append(vector)
    #
    # vectors = sklearn.preprocessing.scale(vectors, axis=1)
    # for resultIndex, result in enumerate(allResults):
    #     for keyIndex, key in enumerate(featureKeys):
    #         result[key] = roundPrecision(vectors[resultIndex][keyIndex], precision=3)

    # Output the normalized results
    # prettyResults = []
    # for result in allResults:
    #     prettyResult = {}
    #     for key in result.keys():
    #         prettyResult[key] = result[key]
    #         if prettyResult[key]:
    #             try:
    #                 number = roundPrecision(float(prettyResult[key]), precision=4)
    #                 prettyResult[key] = number
    #             except ValueError:
    #                 pass
    #             except TypeError:
    #                 pass
    #     prettyResults.append(prettyResult)

    # with open("final/allResults_normalized.csv", "wt") as file:
    #     writer = csv.DictWriter(file, fieldnames=predictorKeys + ['blank1'] + featureKeys + ['blank2'] + nonFeatureKeys)
    #     writer.writeheader()
    #     writer.writerows(prettyResults)

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

    for key in predictorKeys:
        writeDataset(key, "final/" + key + "_testing.csv", testing)
        writeDataset(key, "final/" + key + "_training.csv", training)

    return training, testing


training, testing = mergeResults()


with open('final/allResults.csv', 'rt') as file:
    allResults = list(csv.DictReader(file))

# Now put together the dataset for each predictor feature, so its more convenient to build models on them
shuffled = list(allResults)
random.shuffle(shuffled)
cutoff = int(len(shuffled) * 0.2)
testing = shuffled[:cutoff]
training = shuffled[cutoff:]


def createDataset(key, dataset):
    vectors = []
    targets = []
    allTargets = set()
    for result in dataset:
        if result[key] not in allTargets:
            allTargets.add(result[key])

    allTargets = sorted(list(allTargets))

    for result in dataset:
        if result[key]:
            vector = []
            for feature in featureKeys:
                vector.append(result[feature])
            vectors.append(vector)

            if key in classPredictorKeys:
                targets.append(allTargets.index(result[key]))
            else:
                targets.append(float(result[key]))
    return lightgbm.Dataset(numpy.array(vectors), label=numpy.array(targets), feature_name=featureKeys)


for key in predictorKeys:
    trainingData = createDataset(key, training)
    testingData = createDataset(key, testing)

    params = {
        'num_iterations': 250,
        'is_provide_training_metric': True,
        "early_stopping_round": 3,
        "feature_fraction": 0.70,
        "learning_rate": 0.03
    }

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
        # Now we determine the "adjustmnent factor". Because these models are trained on an extremely noisy data set,
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

        predStddev = numpy.std(trainingPredicted)
        predMean = numpy.mean(trainingPredicted)

        with open('model-' + key + "-configuration.json", 'wt') as file:
            json.dump({
                "origStddev": origStddev,
                "origMean": origStddev,
                "predStddev": predStddev,
                "predMean": predMean,
            }, file)

        def renormalize(value):
            return (((value - predMean) / predStddev) * origStddev) + origMean

        with open('predictions-' + key + '.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=[key, key + "_predicted", key + "_predicted_normalized"])
            writer.writeheader()
            for result in testing:
                if result[key] is not None and result[key] != '':
                    vector = []
                    for feature in featureKeys:
                        vector.append(result[feature])
                    value = model.predict([vector])[0]
                    predicted = renormalize(value)
                    writer.writerow({
                        key: roundPrecision(float(result[key])),
                        key + "_predicted": roundPrecision(float(value)),
                        key + "_predicted_normalized": roundPrecision(float(predicted))
                    })
    else:
        with open('predictions-' + key + '.csv', 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=[key, key + "_predicted"])
            writer.writeheader()
            for result in testing:
                vector = []
                for feature in featureKeys:
                    vector.append(result[feature])

                predicted = atpeParameterValues[key][int(numpy.argmax(model.predict([vector])[0]))]
                writer.writerow({
                    key: result[key],
                    key + "_predicted": predicted
                })


    importances = zip(featureKeys, model.feature_importance())
    importances = sorted(importances, key=lambda r:-r[1])
    print(key)
    for importance in importances:
        print("    ", importance[0], importance[1])
