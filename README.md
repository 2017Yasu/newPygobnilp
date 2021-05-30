# Python implementation of GOBNILP

The Python implementation of GOBNILP is a program that uses Gurobi to
learn Bayesian network structure from either complete discrete data or
complete continuous data or precomputed local scores.

The Python implementation has benefited from work done by Josh Neil on
computing BDeu local scores from discrete data and from work done by
Matt Horder on BGe scoring. In both cases this work was done as a
final-year BEng project at the University of York.

For full details on GOBNILP (including the C version), please consult
the GOBNILP page:

https://www.cs.york.ac.uk/aig/sw/gobnilp/

# Running GOBNILP

To use the Python implementation of GOBNILP it is assumed that you
have Anaconda Python installed as well as Gurobi. One easy way to
achieve this is to install both using [these
instructions](https://www.gurobi.com/get-anaconda/) provided by
Gurobi. You also need to install the numba and pygraphviz Python packages (this be done easily using
Conda). pygraphviz requires [GraphViz](http://www.graphviz.org/) to be installed.
Once you have done all this you just need to grab the Python files
in this git repo and you are good to go.

Here are some example of running the Python implementation using the command line
script rungobnilp.py and 
data files in the data directory. The discrete data
files have the following format: the first line gives the names of the
variables, the second line gives the arity of each variable (i.e. how
many values each can take) and the remaining lines are the data
values. Values are separated by spaces.

Running with default settings:  

`python rungobnilp.py data/asia_10000.dat`

Finding the 4 best BNs (with the default limit, 3, on the size of
parent sets):

`python rungobnilp.py --nopruning --kbest --nsols 4 data/asia_10000.dat`

Finding the 4 best BNs (with the default limit, 3, on the size of
parent sets) and where only one BN for each Markov equivalence class
is allowed:

`python rungobnilp.py --nopruning --mec --kbest --nsols 4 data/asia_10000.dat`

Finding the 4 best BNs with no limit on parent set size and where only
one BN for each Markov equivalence class is allowed:

`python rungobnilp.py --nopruning --mec --kbest --nsols 4 --palim 999 data/asia_10000.dat`

In the examples above where the goal was to find the 'k best' BNs
(subject to various constraints), it was necessary to use
'--nopruning', to turn off pruning. When pruning is used (which is the
default behaviour) then parent sets for BN variables which cannot
occur in an optimal BN are removed, which typically greatly reduces
the size of the problem and speeds up learning.  However, when we seek
not just an optimal BN but also sub-optimal ones then pruning must be
turned off to ensure correct results.


The limit on parent set size is an important parameter. Note that its
default value is 3. Raising this value will slow down learning but may
lead to a higher scoring BN. For example, doing `python
rungobnilp.py --palim 4 data/alarm_100.dat` finds
a higher scoring network than using '--palim 3', and does not take too
long.  Raising to '--palim 5' finds a better (well, higher scoring)
network, but takes just under 100 seconds on my desktop.

The Python implementation of GOBNILP also learns Gaussian networks
from continuous data using BGe scoring. To do this use '--score BGe'
on the command line. (The format for continuous data is similar to that
for discrete data except there is no line for arity.) For example

`python rungobnilp.py --score BGe data/gaussian.dat`

or, to find a higher scoring Gaussian network (with BGe score
-53258.9402):

`python rungobnilp.py --score BGe -p 4 data/gaussian.dat`

The file `gaussian.dat` is from bnlearn where it is called
`guassian.test`. See the [bnlearn
page](http://www.bnlearn.com/documentation/man/gaussian-test.html) for
more information. bnlearn's hillclimbing algorithm `hc` also finds an
optimal network (i.e. with score -53258.9402) using this data. Good
work Marco!


For more details run:
`python rungobnilp.py -h`

