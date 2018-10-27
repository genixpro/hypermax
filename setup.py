from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='hypermax',
    version='0.4.1',
    description='Better, faster hyperparameter optimization by mixing the best of humans and machines.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/electricbrainio/hypermax',
    author='Bradley Arsenault (Electric Brain)',
    author_email='brad@electricbrain.io',
    license='MIT',
    python_requires='>=3',
    packages=find_packages(),
    package_data={
        'hypermax': ['test', 'atpe_models/*.txt', 'atpe_models/*.json'],
    },
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
        'lightgbm',
        'psutil',
        'matplotlib',
        'colors.py'
    ],
    classifiers=[
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    platforms=['Linux', 'OS-X'],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'hypermax = hypermax.cli:main',
        ]
    }
)
