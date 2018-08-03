import argparse
import hypermax.cui
import json
import hyperopt
from pprint import pprint
from hypermax.optimizer import Optimizer

def main():
    parser = argparse.ArgumentParser(description='Provide configuration options for Hypermax')
    parser.add_argument('configuration_file', metavar='configuration_file', type=argparse.FileType('rb'), nargs=1, help='The JSON based configuration file which is used to configure the hyper-parameter search.')

    args = parser.parse_args()

    with args.configuration_file[0] as file:
        config_data = json.load(file)

    optimizer = Optimizer(config_data)

    optimizer.runOptimization()

    hypermax.cui.launchHypermaxUI(optimizer)