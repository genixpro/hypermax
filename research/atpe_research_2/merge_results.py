import os.path
import json
import traceback
import csv
import pickle
from pprint import pprint
from simulation import AlgorithmSimulation

dirs = os.listdir('.')

def roundPrecision(number, precision=3):
    """ Rounds the given floating point number to a certain precision, for output."""
    return float(('{:.' + str(precision) + 'E}').format(number))


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
                    results.append(data)
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
                allResults = allResults + results
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
    for run in logResultsByRun.keys():
        runDuplicates = []
        runAdditionals = []
        for result in logResultsByRun[run]:
            found = False
            if run in csvResultsByRun:
                for result2 in csvResultsByRun[run]:
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
                        # print(run, 'dupes', len(runDuplicates))
                        break
                if not found:
                    runAdditionals.append(result)
                    # print(run, 'adds', len(runAdditionals))

        print(len(runDuplicates))
        print(len(runAdditionals))
        duplicates = duplicates + runDuplicates
        additionals = additionals + runAdditionals

    allResults = csvResults + additionals
    return allResults

def mergeResults():
    allResults = getDeduplicatedResults()

    for result in allResults:
        if 'algorithm' not in result:
            result['algorithm'] = ''
        if 'time' not in result:
            result['time'] = ''

    # Now we can process the results into the final CSV files
    keys = allResults[0].keys()

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
        'secondaryFixedProbability',
        'secondaryLockingMode',
        'secondaryProbabilityMode',
        'secondaryTopLockingPercentile',
    ]

    nonFeatureKeys = ['time', 'fail_rate', 'noise', 'loss', 'algorithm', 'history', 'run']
    for key in keys:
        if 'contribution' in key:
            nonFeatureKeys.append(key)
        if 'interaction' in key:
            nonFeatureKeys.append(key)

    featureKeys = [key for key in keys if key not in predictorKeys and key not in nonFeatureKeys]

    predictorKeys = sorted(predictorKeys)
    nonFeatureKeys = sorted(nonFeatureKeys)
    featureKeys = sorted(featureKeys)

    pprint(predictorKeys)
    pprint(nonFeatureKeys)
    pprint(featureKeys)

    if not os.path.exists('final'):
        os.mkdir('final')
    if not os.path.exists('final/algos'):
        os.mkdir('final/algos')

    savedAlgorithms = {}

    for result in allResults:
        result['blank1'] = ''
        result['blank2'] = ''
        if result['algorithm'] and (result['run'] + result['algorithm']) not in savedAlgorithms:
            data = pickle.load(open(os.path.join(result['run'], 'hypermax', result['algorithm']), 'rb'))['algo']

            scriptName = "algorithm-" + str(len(savedAlgorithms)) + ".py"
            dataName = "algorithm-" + str(len(savedAlgorithms)) + "-pickle.bin"
            with open(os.path.join('final', 'algos', scriptName), 'wt') as file:
                file.write(data.computeScript)
            with open(os.path.join('final', 'algos', dataName), 'wb') as file:
                pickle.dump(data, file)
            savedAlgorithms[result['run'] + result['algorithm']] = scriptName
            result['algorithm'] = scriptName
        elif result['algorithm']:
            result['algorithm'] = savedAlgorithms[result['run'] + result['algorithm']]

    for result in allResults:
        for key in result.keys():
            if result[key]:
                try:
                    number = roundPrecision(float(result[key]), precision=4)
                    result[key] = number
                except ValueError:
                    pass
                except TypeError:
                    pass

    with open("final/allResults.csv", "wt") as file:
        writer = csv.DictWriter(file, fieldnames=predictorKeys + ['blank1'] + featureKeys + ['blank2'] + nonFeatureKeys)
        writer.writeheader()
        writer.writerows(allResults)

    for key in predictorKeys:
        with open('final/' + key + '.csv', 'wt') as file:
            keyResults = []
            for result in allResults:
                if result[key]:
                    data = {}
                    data[key] = result[key]
                    for feature in featureKeys:
                        data[feature] = result[feature]
                    keyResults.append(data)

            writer = csv.DictWriter(file, fieldnames=[key] + featureKeys)
            writer.writeheader()
            writer.writerows(keyResults)

mergeResults()
