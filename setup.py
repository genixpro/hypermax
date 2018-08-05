from setuptools import setup

setup(
    name='hypermax',
    version='0.1',
    description='Better, faster hyperparameter optimization by mixing the best of humans and machines.',
    url='git@github.com:electricbrainio/hypermax.git',
    author='Bradley Arsenault',
    author_email='brad@electricbrain.io',
    license='MIT',
    packages=[
        'hypermax'
    ],
    install_requires=[
        'hyperopt',
        'networkx==1.11',
        'scikit-learn',
        'numpy',
        'scipy',
        'jsonschema',
        'pyyaml',
        'urwid',
        'panwid',
        'psutil',
        'matplotlib',
        'colors.py'
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'hypermax = hypermax.cli:main',
        ]
    }
)
