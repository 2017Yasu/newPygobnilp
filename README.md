# Implementing conditional Gaussian scoring function in pygobnilp and learning Bayesian network from complete mixed data

[Pygobnilp](https://www.cs.york.ac.uk/aig/sw/gobnilp/) is a program developed by James Cussens that supports learning a Bayesian network from complete data. However, this is not able to deal with mixed data, which includes both discrete and continuous values. Therefore, in this project, a method that can evaluate a Bayesian network from a mixed dataset was implemented in pygobnilp, and this made it possible to learn every kind of Bayesian networks.

## Bayesian networks

A Baysian networks (BN) is represented as a *directed acyclic graph (DAG)* G=(V,E) which is a directed graph with the absence of cycles. The following image illustrates an example, called 'Asia', which was introduced by Lauritzen and Spiegelhalter. This represents a probabilistic model for a medical expert system. Each variable can be either TRUE (t) or FALSE (f), and each of them means as follows: A = "visit to Asia", T = "Tuberculosis", X = "Normal X-Ray result", E = "Either tuberculosis or lung cancer", L = "Lung cancer", D = "Dyspnea (shortness of breath)", S = "Smoker", and B = "Bronchitis". It is evident from this BN that variable D is a 'child' of 'parents' E and B, which visually tells that bronchitis and either tuberculosis or lung cancer directly influence the probability to cause dyspnea. In this way, BNs efficiently present probabilistic relationships.

<img width="1440" alt="asia" src="https://user-images.githubusercontent.com/36260690/116341231-7d9da980-a81b-11eb-82d8-1f76dcf04ed4.png">

Pygobnilp would predict the most probable BN by calculating scores of DAGs from the provided dataset.

## Features

The main differences between this new pygobnilp and the original [pygobnilp](https://www.cs.york.ac.uk/aig/sw/gobnilp/) are in [pygobnilp/pygobnilp/scoring.py](https://github.com/2017Yasu/newPygobnilp/blob/main/pygobnilp/pygobnilp/scoring.py). The following classes are added or modified:

- `MixedData` class: This class holds and deals with mixed data. You can call this class by giving `data_source` which is `str`, `array_like`, or `Pandas.DataFrame`, and you can also designate `varnames` and `arities` if you want. For example, you may call `MixedData('sample.txt')`.
- `_AbsLLPenalised` class: This class is an abstract class for calculating penalised log likelihood scores.
- `GaussianLL` class: This class offers calculation of Gaussian LL score which can be used to evaluate a DAG. In this case, the DAG must be learned from a dataset which consists of only continuous data. In particular, newly added method, `score_dag`, calculates the Gaussian score or local Gaussian scores of the given DAG based on the previously given dataset.
- `AbsCGaussianLLScore` class: This is a abstract class for calculating mixed log likelihood scores.
- `CGaussianLL` class: This class offers calculation of conditional Gaussian LL score which can be used to evaluate a DAG. In this case, the DAG must be learned from a dataset which consists of discrete and continuous data.
- `CGaussianBIC` class: This class offers calculation of conditional Gaussian BIC score, which is the log-likelihood penalised by `df * log(N) / 2` for the each pair of child and parents, where `df` is the degrees of freedom of the pair and `N` is the number of variables.
- `CGaussianAIC` class: This class offers calculation of conditional Gaussian AIC score, which is the log-likelihood penalised by `df` for the each pair of child and parents.

A simple example usage can be seen in the [testMixedDataLearning.ipynb](https://github.com/2017Yasu/newPygobnilp/blob/main/testMixedDataLearning.ipynb) notebook. This tutorial uses mixed and continuous dataset extracted from an R package, [bnlearn](https://www.bnlearn.com/).

## Installation

### Dependencies

pygobnilp depends on (1) a number of Python packages (scipy, pygraphviz, matplotlib, networkx, pandas, numpy, scikit-learn and numba) and (2) the Gurobi
MIP solver. pygraphviz also requires [graphviz](https://www.graphviz.org/) to be installed.

Although one can install all these separately the easier option is to install
Anaconda Python and Gurobi together. Just go [here](https://www.gurobi.com/get-anaconda/). Installing Anaconda will get you most of the required
packages but not (at present) pygraphviz, which, once Anaconda is in place,
you can install with:
```
conda install pygraphviz
```
graphviz is not a Python package and has to be installed separately (if you do not already have it on your system).

Gurobi is a commercial system and requires a licence to run. However, an
academic licence is free, see https://www.gurobi.com/academia/academic-program-and-licenses/.
Although you can use pygobnilp with restricted license, the output differs from when using academic license.

### Installing pygobnilp

One installation option is to download this repository and run the following command:
```
python pygobnilp/setup.py develop
```
Full documentation is also available in pygobnilp/\_build/html/index.html.
The original source code of pygobnilp can be found [here](https://bitbucket.org/jamescussens/pygobnilp).

## ???????????????

????????????????????????????????????????????? doc ???????????????????????????????????????????????????????????????????????????
