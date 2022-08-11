from setuptools import find_packages, setup

install_requires = [
    'tqdm',
    'scipy',
    'numpy',
    'tabulate',
    'pandas',
    'scikit_learn>=0.21.0',
    'networkx>=2.3',
    'gensim>=3.8.0',
    'numba>=0.46.0',
]

setup(
    name='graphattack',
    version='0.1.0',
    install_requires=install_requires,
)
