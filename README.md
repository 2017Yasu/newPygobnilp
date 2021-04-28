# Implementing conditional Gaussian scoring function in pygobnilp and learning Bayesian network from complete mixed data

[Pygobnilp](https://www.cs.york.ac.uk/aig/sw/gobnilp/) is a program developed by James Cussens that supports learning a Bayesian network from complete data. However, this is not able to deal with mixed data, which includes both discrete and continuous values. Therefore, in this project, a method that can evaluate a Bayesian network from a mixed dataset was implemented in pygobnilp, and this made it possible to learn every kind of Bayesian networks.

The main differences between this new pygobnilp and the original [pygobnilp](https://www.cs.york.ac.uk/aig/sw/gobnilp/) are in [pygobnilp/pygobnilp/scoring.py](https://github.com/2017Yasu/newPygobnilp/blob/main/pygobnilp/pygobnilp/scoring.py). New classes and methods (such as MixedData and CGaussianLL classes) are added.
