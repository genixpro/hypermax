import argparse
import hypermax.cui
import json
import time
import os.path
import hyperopt
from pprint import pprint
from hypermax.optimizer import Optimizer
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Provide configuration options for Hypermax')
    parser.add_argument('configuration_file', metavar='configuration_file', type=str, nargs=1, help='The JSON based configuration file which is used to configure the hyper-parameter search.')
    parser.add_argument('results_directory', metavar='results_directory', type=str, nargs='?', help='The directory of your existing results to reload and restart from.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    conf = args.configuration_file[0]
    dir_res = args.results_directory
    
    execute(conf, dir_res)
    
def execute(conf, dir_res=None):
    with open(conf) as file:
        config_data = json.load(file)

    optimizer = Optimizer(config_data)

    if dir_res:
        results_path = os.path.join(dir_res, 'results.csv')
        if os.path.exists(results_path):
            optimizer.importResultsCSV(results_path)

        guidance_path = os.path.join(dir_res, 'guidance.json')
        if os.path.exists(guidance_path):
            optimizer.importGuidanceJSON(guidance_path)

        optimizer.resultsAnalyzer.directory = dir_res
    else:
        # See if we see the results directory here.
        directories = os.listdir('.')
        resultsDirectories = sorted([directory for directory in directories if directory.startswith('results_')], key=lambda dir: int(dir[len('result_')+1:]))
        resultsDirectories.reverse() # Reversed - examine the latest results directories first
        for directory in resultsDirectories:
            if os.path.exists(os.path.join(directory, 'search.json')):
                # Check to see if the configuration file is the same
                config = json.load(open(os.path.join(directory, 'search.json'), 'rt'))

                # Compare the config json string (in canonical, sorted form) with the one we've received from the user. If the same, we recommend to the user that they continue with this search
                if json.dumps(config_data, sort_keys=True) == json.dumps(config, sort_keys=True):
                    prompt = input('It appears there was already an in-progress search with this configuration. Would you like to continue the existing hyper parameter search (' + directory + ")? [yes/no/y/n]\n")
                    if 'y' in prompt:
                        optimizer.importResultsCSV(os.path.join(directory, 'results.csv'))
                        if os.path.exists(os.path.join(directory, 'guidance.json')):
                            optimizer.importGuidanceJSON(os.path.join(directory, 'guidance.json'))
                        optimizer.resultsAnalyzer.directory = directory
                    break

    optimizer.runOptimizationThread()

    if config_data.get("ui", {}).get("enabled", True):
        hypermax.cui.launchHypermaxUI(optimizer)

