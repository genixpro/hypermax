from setuptools import setup

setup(
    name='hypermax',
    version='0.1',
    description='Hyper parameter optimization on steroids',
    url='git@github.com:electricbrainio/hypermax.git',
    author='Bradley Arsenault',
    author_email='brad@electricbrain.io',
    license='MIT',
    packages=[
        'hypermax'
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'hypermax = hypermax.cui:launchHypermaxUI',
        ]
    }
)
