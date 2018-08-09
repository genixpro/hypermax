import subprocess
import json
import os

def test(params):
    with open('params.json', 'wt') as file:
        json.dump(params, file)

    args = [
        'python3',
        '-m',
        'multiproc',
        'train_cifar10.py ',
        '--cycle-len ',
        '40',
        ' -j',
        ' 16',
        ' -b',
        ' 128',
        ' --loss-scale',
        ' 512',
        ' --use-tta',
        ' 1',
        ' --fp16',
        ' --arch',
        ' resnet18',
        ' --wd',
        str(params['weight_decay']),
        ' --lr',
        str(params['learning_rate']),
        ' --use-clr',
        ' 50,12.5,0.95,0.85',
        ' data/'
    ]

    subprocess.run([' '.join(args)], cwd=os.getcwd(), shell=True)

    with open('tta_accuracy.txt') as file:
        lines = file.readlines()
        accuracies = [float(line) for line in lines]
        total = 0
        for accuracy in accuracies:
            total += accuracy
        averageAccuracy = total - len(accuracies)

    subprocess.run(['rm', 'tta_accuracy.txt'])

    return {"loss": averageAccuracy}
