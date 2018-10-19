from .optimization_algorithm_base import OptimizationAlgorithmBase
import hyperopt
import functools
import random
import numpy
import numpy.random
import json
import pkg_resources
import tempfile
from hypermax.hyperparameter import Hyperparameter
import sklearn
import lightgbm
import scipy.stats
import math
import copy

class ATPEOptimizer(OptimizationAlgorithmBase):
    atpeParameters = [
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

    atpeParameterCascadeOrdering = [
        'resultFilteringMode',
        'secondaryProbabilityMode',
        'secondaryLockingMode',
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

    atpeParameterValues = {
        'resultFilteringMode': ['age', 'loss_rank', 'none', 'random'],
        'secondaryLockingMode': ['random', 'top'],
        'secondaryProbabilityMode': ['correlation', 'fixed']
    }

    atpeModelFeatureKeys = [
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

    def __init__(self):
        scalingModelData = json.loads(pkg_resources.resource_string(__name__, "../atpe_models/scaling_model.json"))
        self.featureScalingModels = {}
        for key in self.atpeModelFeatureKeys:
            self.featureScalingModels[key] = sklearn.preprocessing.StandardScaler()
            self.featureScalingModels[key].scale_ = numpy.array(scalingModelData[key]['scales'])
            self.featureScalingModels[key].mean_ = numpy.array(scalingModelData[key]['means'])
            self.featureScalingModels[key].var_ = numpy.array(scalingModelData[key]['variances'])

        self.parameterModels = {}
        self.parameterModelConfigurations = {}
        for param in self.atpeParameters:
            modelData = pkg_resources.resource_string(__name__, "../atpe_models/model-" + param + '.txt')
            with tempfile.NamedTemporaryFile() as file:
                file.write(modelData)
                self.parameterModels[param] = lightgbm.Booster(model_file=file.name)

            configString = pkg_resources.resource_string(__name__, "../atpe_models/model-" + param + '-configuration.json')
            data = json.loads(configString)
            self.parameterModelConfigurations[param] = data

        self.lastATPEParameters = None
        self.atpeParamDetails = None


    def recommendNextParameters(self, hyperparameterSpace, results, lockedValues=None):
        rstate = numpy.random.RandomState(seed=int(random.randint(1, 2 ** 32 - 1)))

        params = {}
        def sample(parameters):
            nonlocal params
            params = parameters
            return {"loss": 0.5, 'status': 'ok'}

        parameters = Hyperparameter(hyperparameterSpace).getFlatParameters()

        if lockedValues is not None:
            # Remove any locked values from ones the optimizer will examine
            parameters = list(filter(lambda param: param.name not in lockedValues.keys(), parameters))

        initializationRounds = 10

        atpeParams = {}
        atpeParamDetails = {}
        if len(list(result for result in results if result['loss'])) < initializationRounds:
            atpeParams = {
                'gamma': 1.0,
                'nEICandidates': 24,
                'resultFilteringAgeMultiplier': None,
                'resultFilteringLossRankMultiplier': None,
                'resultFilteringMode': "none",
                'resultFilteringRandomProbability': None,
                'secondaryCorrelationExponent': 1.0,
                'secondaryCorrelationMultiplier': None,
                'secondaryCutoff': 0,
                'secondarySorting': 0,
                'secondaryFixedProbability': 0.5,
                'secondaryLockingMode': 'top',
                'secondaryProbabilityMode': 'fixed',
                'secondaryTopLockingPercentile': 0
            }
        else:
            # Calculate the statistics for the distribution
            stats = self.computeAllResultStatistics(hyperparameterSpace, results)
            stats['num_parameters'] = len(parameters)
            stats['log10_cardinality'] = Hyperparameter(hyperparameterSpace).getLog10Cardinality()
            stats['log10_trial'] = math.log10(len(results))
            baseVector = []

            for feature in self.atpeModelFeatureKeys:
                scalingModel = self.featureScalingModels[feature]
                transformed = scalingModel.transform([[stats[feature]]])[0][0]
                baseVector.append(transformed)

            baseVector = numpy.array([baseVector])

            for atpeParamIndex, atpeParameter in enumerate(self.atpeParameterCascadeOrdering):
                vector = copy.copy(baseVector)[0].tolist()
                atpeParamFeatures = self.atpeParameterCascadeOrdering[:atpeParamIndex]
                for atpeParamFeature in atpeParamFeatures:
                    # We have to insert a special value of -3 for any conditional parameters.
                    if atpeParamFeature == 'resultFilteringAgeMultiplier' and atpeParams['resultFilteringMode'] != 'age':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'resultFilteringLossRankMultiplier' and atpeParams['resultFilteringMode'] != 'loss_rank':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'resultFilteringRandomProbability' and atpeParams['resultFilteringMode'] != 'random':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'secondaryCorrelationMultiplier' and atpeParams['secondaryProbabilityMode'] != 'correlation':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'secondaryFixedProbability' and atpeParams['secondaryProbabilityMode'] != 'fixed':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature == 'secondaryTopLockingPercentile' and atpeParams['secondaryLockingMode'] != 'top':
                        vector.append(-3)  # This is the default value inserted when parameters aren't relevant
                    elif atpeParamFeature in self.atpeParameterValues:
                        for value in self.atpeParameterValues[atpeParamFeature]:
                            vector.append(1.0 if atpeParams[atpeParamFeature] == value else 0)
                    else:
                        vector.append(float(atpeParams[atpeParamFeature]))

                allFeatureKeysForATPEParamModel = copy.copy(self.atpeModelFeatureKeys)
                for atpeParamFeature in atpeParamFeatures:
                    if atpeParamFeature in self.atpeParameterValues:
                        for value in self.atpeParameterValues[atpeParamFeature]:
                            allFeatureKeysForATPEParamModel.append(atpeParamFeature + "_" + value)
                    else:
                        allFeatureKeysForATPEParamModel.append(atpeParamFeature)

                value = self.parameterModels[atpeParameter].predict([vector])[0]
                featureContributions = self.parameterModels[atpeParameter].predict([vector], pred_contrib=True)[0]

                atpeParamDetails[atpeParameter] = {
                    "value": None,
                    "reason": None
                }

                # Set the value
                if atpeParameter in self.atpeParameterValues:
                    # Renormalize the predicted probabilities
                    config = self.parameterModelConfigurations[atpeParameter]
                    for atpeParamValueIndex, atpeParamValue in enumerate(self.atpeParameterValues[atpeParameter]):
                        value[atpeParamValueIndex] = (((value[atpeParamValueIndex] - config['predMeans'][atpeParamValue]) / config['predStddevs'][atpeParamValue]) *
                                                      config['origStddevs'][atpeParamValue]) + config['origMeans'][atpeParamValue]
                        value[atpeParamValueIndex] = max(0.0, min(1.0, value[atpeParamValueIndex]))

                    maxVal = numpy.max(value)
                    for atpeParamValueIndex, atpeParamValue in enumerate(self.atpeParameterValues[atpeParameter]):
                        value[atpeParamValueIndex] = max(value[atpeParamValueIndex], maxVal * 0.15)  # We still allow the non reccomended modes to get chosen 15% of the time

                    # Make a random weighted choice based on the normalized probabilities
                    probabilities = value / numpy.sum(value)
                    chosen = numpy.random.choice(a=self.atpeParameterValues[atpeParameter], p=probabilities)
                    atpeParams[atpeParameter] = str(chosen)
                else:
                    # Renormalize the predictions
                    config = self.parameterModelConfigurations[atpeParameter]
                    value = (((value - config['predMean']) / config['predStddev']) * config['origStddev']) + config['origMean']
                    atpeParams[atpeParameter] = float(value)

                atpeParamDetails[atpeParameter]["reason"] = {}
                # If we are predicting a class, we get separate feature contributions for each class. Take the average
                if atpeParameter in self.atpeParameterValues:
                    featureContributions = numpy.mean(
                        numpy.reshape(featureContributions, newshape=(len(allFeatureKeysForATPEParamModel) + 1, len(self.atpeParameterValues[atpeParameter]))), axis=1)

                contributions = [(self.atpeModelFeatureKeys[index], float(featureContributions[index])) for index in range(len(self.atpeModelFeatureKeys))]
                contributions = sorted(contributions, key=lambda r: -r[1])
                # Only focus on the top 10% of features, since it gives more useful information. Otherwise the total gets really squashed out over many features,
                # because our model is highly regularized.
                contributions = contributions[:int(len(contributions) / 10)]
                total = numpy.sum([contrib[1] for contrib in contributions])

                for contributionIndex, contribution in enumerate(contributions[:3]):
                    atpeParamDetails[atpeParameter]['reason'][contribution[0]] = str(int(float(contribution[1]) * 100.0 / total)) + "%"

                # Apply bounds to all the parameters
                if atpeParameter == 'gamma':
                    atpeParams['gamma'] = max(0.2, min(2.0, atpeParams['gamma']))
                if atpeParameter == 'nEICandidates':
                    atpeParams['nEICandidates'] = int(max(2.0, min(48, atpeParams['nEICandidates'])))
                if atpeParameter == 'resultFilteringAgeMultiplier':
                    atpeParams['resultFilteringAgeMultiplier'] = max(1.0, min(4.0, atpeParams['resultFilteringAgeMultiplier']))
                if atpeParameter == 'resultFilteringLossRankMultiplier':
                    atpeParams['resultFilteringLossRankMultiplier'] = max(1.0, min(4.0, atpeParams['resultFilteringLossRankMultiplier']))
                if atpeParameter == 'resultFilteringRandomProbability':
                    atpeParams['resultFilteringRandomProbability'] = max(0.7, min(0.9, atpeParams['resultFilteringRandomProbability']))
                if atpeParameter == 'secondaryCorrelationExponent':
                    atpeParams['secondaryCorrelationExponent'] = max(1.0, min(3.0, atpeParams['secondaryCorrelationExponent']))
                if atpeParameter == 'secondaryCorrelationMultiplier':
                    atpeParams['secondaryCorrelationMultiplier'] = max(0.2, min(1.8, atpeParams['secondaryCorrelationMultiplier']))
                if atpeParameter == 'secondaryCutoff':
                    atpeParams['secondaryCutoff'] = max(-1.0, min(1.0, atpeParams['secondaryCutoff']))
                if atpeParameter == 'secondaryFixedProbability':
                    atpeParams['secondaryFixedProbability'] = max(0.2, min(0.8, atpeParams['secondaryFixedProbability']))
                if atpeParameter == 'secondaryTopLockingPercentile':
                    atpeParams['secondaryTopLockingPercentile'] = max(0, min(10.0, atpeParams['secondaryTopLockingPercentile']))

            # Now blank out unneeded params so they don't confuse us
            if atpeParams['secondaryLockingMode'] == 'random':
                atpeParams['secondaryTopLockingPercentile'] = None

            if atpeParams['secondaryProbabilityMode'] == 'fixed':
                atpeParams['secondaryCorrelationMultiplier'] = None
            else:
                atpeParams['secondaryFixedProbability'] = None

            if atpeParams['resultFilteringMode'] == 'none':
                atpeParams['resultFilteringAgeMultiplier'] = None
                atpeParams['resultFilteringLossRankMultiplier'] = None
                atpeParams['resultFilteringRandomProbability'] = None
            elif atpeParams['resultFilteringMode'] == 'age':
                atpeParams['resultFilteringLossRankMultiplier'] = None
                atpeParams['resultFilteringRandomProbability'] = None
            elif atpeParams['resultFilteringMode'] == 'loss_rank':
                atpeParams['resultFilteringAgeMultiplier'] = None
                atpeParams['resultFilteringRandomProbability'] = None
            elif atpeParams['resultFilteringMode'] == 'random':
                atpeParams['resultFilteringAgeMultiplier'] = None
                atpeParams['resultFilteringLossRankMultiplier'] = None

            for atpeParameter in self.atpeParameters:
                if atpeParams[atpeParameter] is None:
                    del atpeParamDetails[atpeParameter]
                else:
                    atpeParamDetails[atpeParameter]['value'] = atpeParams[atpeParameter]

        self.lastATPEParameters = atpeParams
        self.atpeParamDetails = atpeParamDetails

        # pprint(atpeParams)

        def computePrimarySecondary():
            if len(results) < initializationRounds:
                return parameters, [], [0.5] * len(parameters)  # Put all parameters as primary

            if len(set(result['loss'] for result in results)) < 5:
                return parameters, [], [0.5] * len(parameters)  # Put all parameters as primary

            numberParameters = [parameter for parameter in parameters if parameter.config['type'] == 'number']
            otherParameters = [parameter for parameter in parameters if parameter.config['type'] != 'number']

            totalWeight = 0
            correlations = {}
            for parameter in numberParameters:
                if len(set(result[parameter.name] for result in results if result[parameter.name] is not None)) < 2:
                    correlations[parameter.name] = 0
                else:
                    values = []
                    valueLosses = []
                    for result in results:
                        if result[parameter.name] is not None and result['loss'] is not None:
                            values.append(result[parameter.name])
                            valueLosses.append(result['loss'])

                    correlation = math.pow(abs(scipy.stats.spearmanr(values, valueLosses)[0]), atpeParams['secondaryCorrelationExponent'])
                    correlations[parameter.name] = correlation
                    totalWeight += correlation

            threshold = totalWeight * abs(atpeParams['secondaryCutoff'])

            if atpeParams['secondaryCutoff'] < 0:
                # Reverse order - we lock in the highest correlated parameters
                sortedParameters = sorted(numberParameters, key=lambda parameter: correlations[parameter.name])
            else:
                # Normal order - sort properties by their correlation to lock in lowest correlated parameters
                sortedParameters = sorted(numberParameters, key=lambda parameter: -correlations[parameter.name])

            primaryParameters = []
            secondaryParameters = []
            cumulative = totalWeight
            for parameter in sortedParameters:
                if cumulative < threshold:
                    secondaryParameters.append(parameter)
                else:
                    primaryParameters.append(parameter)

                cumulative -= correlations[parameter.name]

            return primaryParameters + otherParameters, secondaryParameters, correlations

        if len([result['loss'] for result in results if result['loss'] is not None]) == 0:
            maxLoss = 1
        else:
            maxLoss = numpy.max([result['loss'] for result in results if result['loss'] is not None])

        # We create a copy of lockedValues so we don't modify the object that was passed in as an argument - treat it as immutable.
        # The ATPE algorithm will lock additional values in a stochastic manner
        if lockedValues is None:
            lockedValues = {}
        else:
            lockedValues = copy.copy(lockedValues)

        filteredResults = []
        removedResults = []
        if len(results) > initializationRounds:
            primaryParameters, secondaryParameters, correlations = computePrimarySecondary()

            sortedResults = list(sorted(list(results), key=lambda result: (result['loss'] if result['loss'] is not None else (maxLoss + 1))))
            topResults = sortedResults
            if atpeParams['secondaryLockingMode'] == 'top':
                topResultsN = max(1, int(math.ceil(len(sortedResults) * atpeParams['secondaryTopLockingPercentile'] / 100.0)))
                topResults = sortedResults[:topResultsN]

            # Any secondary parameters have may be locked to either the current best value or any value within the result pool.
            for secondary in secondaryParameters:
                if atpeParams['secondaryProbabilityMode'] == 'fixed':
                    if random.uniform(0, 1) < atpeParams['secondaryFixedProbability']:
                        if atpeParams['secondaryLockingMode'] == 'top':
                            lockResult = random.choice(topResults)
                            if lockResult[secondary.name] is not None:
                                lockedValues[secondary.name] = lockResult[secondary.name]
                        elif atpeParams['secondaryLockingMode'] == 'random':
                            minVal = secondary.config['min']
                            maxVal = secondary.config['max']

                            if secondary.config.get('scaling', 'linear') == 'logarithmic':
                                minVal = math.log(minVal)
                                maxVal = math.log(maxVal)

                            value = random.uniform(minVal, maxVal)

                            if secondary.config.get('scaling', 'linear') == 'logarithmic':
                                value = math.exp(value)

                            if 'rounding' in secondary.config:
                                value = round(value / secondary.config['rounding']) * secondary.config['rounding']

                            lockedValues[secondary.name] = value

                elif atpeParams['secondaryProbabilityMode'] == 'correlation':
                    probability = max(0, min(1, abs(correlations[secondary.name]) * atpeParams['secondaryCorrelationMultiplier']))
                    if random.uniform(0, 1) < probability:
                        if atpeParams['secondaryLockingMode'] == 'top':
                            lockResult = random.choice(topResults)
                            if lockResult[secondary.name] is not None:
                                lockedValues[secondary.name] = lockResult[secondary.name]
                        elif atpeParams['secondaryLockingMode'] == 'random':
                            minVal = secondary.config['min']
                            maxVal = secondary.config['max']

                            if secondary.config.get('scaling', 'linear') == 'logarithmic':
                                minVal = math.log(minVal)
                                maxVal = math.log(maxVal)

                            value = random.uniform(minVal, maxVal)

                            if secondary.config.get('scaling', 'linear') == 'logarithmic':
                                value = math.exp(value)

                            if 'rounding' in secondary.config:
                                value = round(value / secondary.config['rounding']) * secondary.config['rounding']

                            lockedValues[secondary.name] = value

            # Now last step, we filter results prior to sending them into ATPE
            for resultIndex, result in enumerate(results):
                if atpeParams['resultFilteringMode'] == 'none':
                    filteredResults.append(result)
                elif atpeParams['resultFilteringMode'] == 'random':
                    if random.uniform(0, 1) < atpeParams['resultFilteringRandomProbability']:
                        filteredResults.append(result)
                    else:
                        removedResults.append(result)
                elif atpeParams['resultFilteringMode'] == 'age':
                    age = float(resultIndex) / float(len(results))
                    if random.uniform(0, 1) < (atpeParams['resultFilteringAgeMultiplier'] * age):
                        filteredResults.append(result)
                    else:
                        removedResults.append(result)
                elif atpeParams['resultFilteringMode'] == 'loss_rank':
                    rank = 1.0 - (float(sortedResults.index(result)) / float(len(results)))
                    if random.uniform(0, 1) < (atpeParams['resultFilteringLossRankMultiplier'] * rank):
                        filteredResults.append(result)
                    else:
                        removedResults.append(result)

        # If we are in initialization, or by some other fluke of random nature that we end up with no results after filtering,
        # then just use all the results
        if len(filteredResults) == 0:
            filteredResults = results

        hyperopt.fmin(fn=sample,
                      space=Hyperparameter(hyperparameterSpace).createHyperoptSpace(lockedValues),
                      algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=initializationRounds, gamma=atpeParams['gamma'],
                                             n_EI_candidates=int(atpeParams['nEICandidates'])),
                      max_evals=1,
                      trials=self.convertResultsToTrials(hyperparameterSpace, filteredResults),
                      rstate=rstate)
        
        return params



    def computePartialResultStatistics(self, hyperparameterSpace, results):
        losses = numpy.array(sorted([result['loss'] for result in results if result['loss'] is not None]))

        bestLoss = 0
        percentile5Loss = 0
        percentile25Loss = 0
        percentile50Loss = 0
        percentile75Loss = 0
        statistics = {}

        numpy.warnings.filterwarnings('ignore')

        if len(set(losses)) > 1:
            bestLoss = numpy.percentile(losses, 0)
            percentile5Loss = numpy.percentile(losses, 5)
            percentile25Loss = numpy.percentile(losses, 25)
            percentile50Loss = numpy.percentile(losses, 50)
            percentile75Loss = numpy.percentile(losses, 75)

            statistics['loss_skew'] = scipy.stats.skew(losses)
            statistics['loss_kurtosis'] = scipy.stats.kurtosis(losses)
        else:
            statistics['loss_skew'] = 0
            statistics['loss_kurtosis'] = 0

        if percentile50Loss == 0:
            statistics['loss_stddev_median_ratio'] = 0
            statistics['loss_best_percentile50_ratio'] = 0
        else:
            statistics['loss_stddev_median_ratio'] = numpy.std(losses) / percentile50Loss
            statistics['loss_best_percentile50_ratio'] = bestLoss / percentile50Loss

        if bestLoss == 0:
            statistics['loss_stddev_best_ratio'] = 0
        else:
            statistics['loss_stddev_best_ratio'] = numpy.std(losses) / bestLoss

        if percentile25Loss == 0:
            statistics['loss_best_percentile25_ratio'] = 0
            statistics['loss_percentile5_percentile25_ratio'] = 0
        else:
            statistics['loss_best_percentile25_ratio'] = bestLoss / percentile25Loss
            statistics['loss_percentile5_percentile25_ratio'] = percentile5Loss / percentile25Loss

        if percentile75Loss == 0:
            statistics['loss_best_percentile75_ratio'] = 0
        else:
            statistics['loss_best_percentile75_ratio'] = bestLoss / percentile75Loss

        def getValue(result, parameter):
            return result[parameter.name]

        # Now we compute correlations between each parameter and the loss
        parameters = Hyperparameter(hyperparameterSpace).getFlatParameters()
        correlations = []
        for parameter in parameters:
            if parameter.config['type'] == 'number':
                if len(set(getValue(result, parameter) for result in results if (getValue(result, parameter) is not None and result['loss'] is not None))) < 2:
                    correlations.append(0)
                else:
                    values = []
                    valueLosses = []
                    for result in results:
                        if result['loss'] is not None and (isinstance(getValue(result, parameter), float) or isinstance(getValue(result, parameter), int)):
                            values.append(getValue(result, parameter))
                            valueLosses.append(result['loss'])

                    correlation = abs(scipy.stats.spearmanr(values, valueLosses)[0])
                    if math.isnan(correlation) or math.isinf(correlation):
                        correlations.append(0)
                    else:
                        correlations.append(correlation)

        correlations = numpy.array(correlations)

        if len(set(correlations)) == 1:
            statistics['correlation_skew'] = 0
            statistics['correlation_kurtosis'] = 0
            statistics['correlation_stddev_median_ratio'] = 0
            statistics['correlation_stddev_best_ratio'] = 0

            statistics['correlation_best_percentile25_ratio'] = 0
            statistics['correlation_best_percentile50_ratio'] = 0
            statistics['correlation_best_percentile75_ratio'] = 0
            statistics['correlation_percentile5_percentile25_ratio'] = 0
        else:
            bestCorrelation = numpy.percentile(correlations, 100) # Correlations are in the opposite order of losses, higher correlation is considered "best"
            percentile5Correlation = numpy.percentile(correlations, 95)
            percentile25Correlation = numpy.percentile(correlations, 75)
            percentile50Correlation = numpy.percentile(correlations, 50)
            percentile75Correlation = numpy.percentile(correlations, 25)

            statistics['correlation_skew'] = scipy.stats.skew(correlations)
            statistics['correlation_kurtosis'] = scipy.stats.kurtosis(correlations)

            if percentile50Correlation == 0:
                statistics['correlation_stddev_median_ratio'] = 0
                statistics['correlation_best_percentile50_ratio'] = 0
            else:
                statistics['correlation_stddev_median_ratio'] = numpy.std(correlations) / percentile50Correlation
                statistics['correlation_best_percentile50_ratio'] = bestCorrelation / percentile50Correlation

            if bestCorrelation == 0:
                statistics['correlation_stddev_best_ratio'] = 0
            else:
                statistics['correlation_stddev_best_ratio'] = numpy.std(correlations) / bestCorrelation

            if percentile25Correlation == 0:
                statistics['correlation_best_percentile25_ratio'] = 0
                statistics['correlation_percentile5_percentile25_ratio'] = 0
            else:
                statistics['correlation_best_percentile25_ratio'] = bestCorrelation / percentile25Correlation
                statistics['correlation_percentile5_percentile25_ratio'] = percentile5Correlation / percentile25Correlation

            if percentile75Correlation == 0:
                statistics['correlation_best_percentile75_ratio'] = 0
            else:
                statistics['correlation_best_percentile75_ratio'] = bestCorrelation / percentile75Correlation

        return statistics

    def computeAllResultStatistics(self, hyperparameterSpace, results):
        losses = numpy.array(sorted([result['loss'] for result in results if result['loss'] is not None]))

        if len(set(losses)) > 1:
            percentile10Loss = numpy.percentile(losses, 10)
            percentile20Loss = numpy.percentile(losses, 20)
            percentile30Loss = numpy.percentile(losses, 30)
        else:
            percentile10Loss = losses[0]
            percentile20Loss = losses[0]
            percentile30Loss = losses[0]

        allResults = list(results)
        percentile10Results = [result for result in results if result['loss'] is not None and result['loss'] <= percentile10Loss]
        percentile20Results = [result for result in results if result['loss'] is not None and result['loss'] <= percentile20Loss]
        percentile30Results = [result for result in results if result['loss'] is not None and result['loss'] <= percentile30Loss]

        recent10Count = min(len(results), 10)
        recent10Results = results[-recent10Count:]

        recent25Count = min(len(results), 25)
        recent25Results = results[-recent25Count:]

        recent15PercentCount = max(math.ceil(len(results)*0.15), 5)
        recent15PercentResults = results[-recent15PercentCount:]

        statistics = {}
        allResultStatistics = self.computePartialResultStatistics(hyperparameterSpace, allResults)
        for stat,value in allResultStatistics.items():
            statistics['all_' + stat] = value

        percentile10Statistics = self.computePartialResultStatistics(hyperparameterSpace, percentile10Results)
        for stat,value in percentile10Statistics.items():
            statistics['top_10%_' + stat] = value

        percentile20Statistics = self.computePartialResultStatistics(hyperparameterSpace, percentile20Results)
        for stat,value in percentile20Statistics.items():
            statistics['top_20%_' + stat] = value

        percentile30Statistics = self.computePartialResultStatistics(hyperparameterSpace, percentile30Results)
        for stat,value in percentile30Statistics.items():
            statistics['top_30%_' + stat] = value

        recent10Statistics = self.computePartialResultStatistics(hyperparameterSpace, recent10Results)
        for stat,value in recent10Statistics.items():
            statistics['recent_10_' + stat] = value

        recent25Statistics = self.computePartialResultStatistics(hyperparameterSpace, recent25Results)
        for stat,value in recent25Statistics.items():
            statistics['recent_25_' + stat] = value

        recent15PercentResult = self.computePartialResultStatistics(hyperparameterSpace, recent15PercentResults)
        for stat,value in recent15PercentResult.items():
            statistics['recent_15%_' + stat] = value

        # Although we have added lots of protection in the computePartialResultStatistics code, one last hedge against any NaN or infinity values coming up
        # in our statistics
        for key in statistics.keys():
            if math.isnan(statistics[key]) or math.isinf(statistics[key]):
                statistics[key] = 0

        return statistics
