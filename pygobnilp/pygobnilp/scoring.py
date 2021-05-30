#/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *   GOBNILP (Python version) Copyright (C) 2019 James Cussens           *
# *                                                                       *
# *   This program is free software; you can redistribute it and/or       *
# *   modify it under the terms of the GNU General Public License as      *
# *   published by the Free Software Foundation; either version 3 of the  *
# *   License, or (at your option) any later version.                     *
# *                                                                       *
# *   This program is distributed in the hope that it will be useful,     *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of      *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU    *
# *   General Public License for more details.                            *
# *                                                                       *
# *   You should have received a copy of the GNU General Public License   *
# *   along with this program; if not, see                                *
# *   <http://www.gnu.org/licenses>.                                      *
"""
    Code for computing local scores from data
"""

__author__ = "Josh Neil, James Cussens"
__email__ = "james.cussens@york.ac.uk"

from math import lgamma, log, pi
from itertools import combinations

from scipy.special import digamma, gammaln
from scipy.stats import norm, entropy

import numpy as np
import pandas as pd

from numba import jit, njit

# try:
#     from .adtree import return_adtree, makecontab
#     adtree_available = True
# except ImportErorr as e:
#     adtree_available = False
#     print("C ADTree implementation unavailable.")
#     print(e)

adtree_available = False
    
# START functions for contabs

@jit(nopython=True)
def marginalise_uniques_contab(unique_insts, counts, cols):
    '''
    Marginalise a contingency table represented by unique insts and counts
    '''
    marg_uniqs, indices = np.unique(unique_insts[:,cols], return_inverse=True)
    marg_counts = np.zeros(len(marg_uniqs),dtype=np.uint32)
    for i in range(len(counts)):
        marg_counts[indices[i]] += counts[i]
    return marg_uniqs, marg_counts

@jit(nopython=True)
def make_contab(data, counts, cols, arities, maxsize):
    '''
    Compute a marginal contingency table from data or report
    that the desired contingency table would be too big.

    All inputs except the last are arrays of unsigned integers

    Args:
     data (numpy array): the unique datapoints as a 2-d array, each row is a datapoint, assumed unique
     counts (numpy array): the count of how often each unique datapoint occurs in the original data
     cols (numpy array): the columns (=variables) for the marginal contingency table.
      columns must be ordered low to high
     arities (numpy array): the arities of the variables (=columns) for the contingency table
      order must match that of `cols`.
     maxsize (int): the maximum size (number of cells) allowed for a contingency table

    Returns:
     tuple: 1st element is of type ndarray: 
      If the contingency table would have more elements than `maxsize' then the array is empty
      (and the 2nd element of the tuple should be ignored)
      else an array of counts of length equal to the product of the `arities`.
      Counts are in lexicographic order of the joint instantiations of the columns (=variables)
      2nd element: the 'strides' for each column (=variable)
    '''
    p = len(cols)
    #if arities = (2,3,3) then strides = 9,3,1
    #if row is (2,1,2) then index is 2*9 + 1*3 + 2*1
    strides = np.empty(p,dtype=np.uint32)
    idx = p-1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[idx]
        if stride > maxsize:
            return np.empty(0,dtype=np.uint32), strides
        idx -= 1
    contab = np.zeros(stride,dtype=np.uint32)
    for rowidx in range(data.shape[0]):
        idx = 0
        for i in range(p):
            idx += data[rowidx,cols[i]]*strides[i]
        contab[idx] += counts[rowidx]
    return contab, strides

@jit(nopython=True)
def _compute_ll_from_flat_contab(contab,strides,child_idx,child_arity):
    child_stride = strides[child_idx]
    ll = 0.0
    child_counts = np.empty(child_arity,dtype=np.int32)
    contab_size = len(contab)
    for i in range(0,contab_size,child_arity*child_stride):
        for j in range(i,i+child_stride):
            n = 0
            for k in range(child_arity):
                count = contab[j]
                child_counts[k] = count
                n += count
                j += child_stride
            if n > 0:
                for c in child_counts:
                    if c > 0:
                        ll += c * log(c/n)
    return ll

@jit(nopython=True)
def _compute_ll_from_unique_contab(data,counts,n_uniqs,pa_idxs,child_arity,orig_child_col):
    child_counts = np.zeros((n_uniqs,np.int64(child_arity)),dtype=np.uint32)
    for i in range(len(data)):
        child_counts[pa_idxs[i],data[i,orig_child_col]] += counts[i]
    ll = 0.0
    for i in range(len(child_counts)):
        #n = child_counts[i,:].sum() #to consider
        n = 0
        for k in range(child_arity):
            n += child_counts[i,k]
        for k in range(child_arity):
            c = child_counts[i,k]
            if c > 0:
                ll += c * log(c/n)
    return ll

@jit(nopython=True)
def compute_bdeu_component(data, counts, cols, arities, alpha, maxflatcontabsize):
    contab = make_contab(data, counts, cols, arities, maxflatcontabsize)[0]
    if len(contab) > 0:
        alpha_div_arities = alpha / len(contab)
        non_zero_count = 0
        score = 0.0
        for count in contab:
            if count != 0:
                non_zero_count += 1
                score -= lgamma(alpha_div_arities+count) 
        score += non_zero_count*lgamma(alpha_div_arities)  
        return score, non_zero_count


# START functions for upper bounds

@jit(nopython=True)
def lg(n,x):
    return lgamma(n+x) - lgamma(x)

def h(counts):
    '''
    log P(counts|theta) where theta are MLE estimates computed from counts
    ''' 
    tot = sum(counts)
    #print(tot)
    res = 0.0
    for n in counts:
        if n > 0:
            res += n*log(n/tot)
    return res

def chisq(counts):
    tot = sum(counts)
    t = len(counts)
    mean = tot/t #Python 3 - this creates a float
    chisq = 0.0
    for n in counts:
        chisq += (n - mean)**2
    return chisq/mean

def hsum(distss):
    res = 0.0
    for dists in distss:
        res += sum([h(d) for d in dists])
    return res

@jit(nopython=True)
def fa(dist,sumdist,alpha,r):
    #res = -lg(sumdist,alpha)
    res = lgamma(alpha) - lgamma(sumdist+alpha)
    alphar = alpha/r
    k = 0
    for n in dist:
        if n > 0:
            #res += lg(n,alphar)
            #res += (lgamma(n+alphar) - lgamma(alphar))
            res += lgamma(n+alphar)
            k += 1
    return res - k*lgamma(alphar)

def diffa(dist,alpha,r):
    """Compute the derivative of local-local BDeu score

    numba does not support the scipy.special functions -- we are using the
    digamma function in defining the entropy of the Chisq
    distribution. For this end I added a python script which contains code
    for a @vectorize-d digamma function. If we need to use anything from
    scipy.special we will have to write it up ourselves.

    """
    args = [n+alpha/r for n in dist] + [alpha,sum(dist)+alpha,alpha/r]
    z = digamma(args)
    return sum(z[:r+1]) - z[r+1] - r*z[r+2] 

def onepositive(dist):
    """
    Is there only one positive count in `dist`?
    """
    npos = 0
    for n in dist:
        if n > 0:
            npos += 1
            if npos > 1:
                return False
    return True

@njit
def array_equal(a,b,pasize):
    for i in range(pasize):
        if a[i] != b[i]:
            return False
    return True

#@njit
#def get_elems(a,idxs):
#    return np.a

@njit
def upper_bound_james_fun(atoms_ints,atoms_floats,pasize,alpha,r,idxs):
    ub = 0.0
    local_ub = 0.0
    best_diff = 0.0
    pasize_r = pasize+r
    lr = -log(r)
    oldrow = atoms_ints[idxs[0]]
    for i in idxs:
        row = atoms_ints[i]
        
        if not array_equal(oldrow,row,pasize):
            ub += min(local_ub+best_diff,lr)
            best_diff = 0.0
            local_ub = 0.0
            oldrow = row

        mle = atoms_floats[i]
        local_ub += mle
        if row[-1]: # if chi-sq condition met
            diff = fa(row[pasize:pasize_r],row[pasize_r],alpha/2.0,r) - mle
            if diff < best_diff:
                best_diff = diff
    ub += min(local_ub+best_diff,lr)
    return ub


#@jit(nopython=True)
def ub(dists,alpha,r):
    '''
    Args:
        dists (iter): list of (child-value-counts,chisqtest,mles) lists, for some
          particular instantiation ('inst1') of the current parents.
          There is a child-value-counts list for each non-zero instantiation ('inst2') of the biggest
          possible superset of the current set of parents. inst1 and each inst2 have the same
          values for the current parents.
        alpha (float): ESS/(num of current parent insts)
        r (int): arity of the child

    Returns:
       float: An upper bound on the BDeu local score for (implicit) child
        over all possible proper supersets of the current parents
    '''
    naives = 0.0
    best_diff = 0.0
    for (dist,sumdist,ok_first,naive_ub) in dists:
        naives += naive_ub
        #iffirst_ub = naive_ub
        #iffirst_ub = min(len(dist)*-log(r),naive_ub)
        if ok_first:
            diff = fa(dist,sumdist,alpha/2.0,r) - naive_ub
            #iffirst_ub = min(iffirst_ub,fa(dist,sumdist,alpha/2.0,r))
            #diff = iffirst_ub - naive_ub
            if diff < best_diff:
                best_diff = diff
    return best_diff + naives


# @jit(nopython=True)
# def _ll_score(contab_generator, arities, variables, child, process_contab_function):
#     """
#         Parameters:

#         - `contab_generator`: Function used to generate the contingency tables for `variables`
#         - `arities`: A NumPy array containing the arity of every variable
#         - `variables`: A Numpy array of ints which are column indices for the family = (child + parents)
#         - `child`: The column index for the child 
#         - `process_contab_function`: Function called to do main computation

#         - Returns the fitted log-likelihood (LL) score 
#     """
#     contab = contab_generator.make_contab(variables)
#     score, non_zero_count = process_contab_function(arities, contab, variables, alpha_div_arities)             
    
#     return score, non_zero_count



#@jit(nopython=True)
def get_atoms(data,arities):
    '''
    Args: 
        data(np.array): Discrete data as a 2d array of ints

    Returns:
        list: a list `atoms`, where `atoms[i]` is a dictionary mapping instantations
         of variables other than i to a tuple with 3 elements:

          1. the child counts (n1, n2, .., nr) for that instantations
          2. n * the entropy for the empirical distribution (n1/n, n2/n, .., nr/n)
          3. Whether the chi-squared statistic for (n1, n2, .., nr) exceeds r-1 
        
        Chi-squared test comes from "Compound Multinomial Likelihood Functions are Unimodal: 
        Proof of a Conjecture of I. J. Good". Author(s): Bruce Levin and James Reeds
        Source: The Annals of Statistics, Vol. 5, No. 1 (Jan., 1977), pp. 79-87
    
        At least two of the ni must be positive for the inst-tuple pair to be included
        in the dictionary since only in that case is the inst-tuple useful for getting
        a good upper bound.
       
    '''
    fullinsts = []
    for i in range(data.shape[1]):
        fullinsts.append({})
    for row in data:
        row = tuple(row)
        for i, val in enumerate(row):
            fullinsts[i].setdefault(row[:i]+(0,)+row[i+1:],  # add dummy value for ith val
                                    [0]*arities[i])[val] += 1

    # now, for each child i, delete full insts which are 'deterministic'
    # i.e. where only one child value is non-zero
    newdkts_ints = []
    newdkts_floats = []
    for i, dkt in enumerate(fullinsts):
        #print('old',len(dkt))
        #newdkt = {}
        newdkt_ints = []
        newdkt_floats = []
        th = arities[i] - 1
        for inst, childcounts in dkt.items():
            if len([v for v in childcounts if v > 0]) > 1:
                newdkt_ints.append(inst+tuple(childcounts)+(sum(childcounts),chisq(childcounts) > th))
                #newdkt[inst] = (
                #    np.array(childcounts,np.uint64),sum(childcounts),
                #    chisq(childcounts) > th,
                #    h(childcounts))
                newdkt_floats.append(h(childcounts))
        newdkts_ints.append(np.array(newdkt_ints,np.uint64))
        newdkts_floats.append(np.array(newdkt_floats,np.float64))
        #print('new',len(newdkt))
    #fullinsts = newdkts

    #print(newdkts_ints[0])
    #sys.exit()

    #for x in newdkts_ints:
    #    print(x.shape)
    
    return newdkts_ints, newdkts_floats
    

def save_local_scores(local_scores, filename):
    variables = local_scores.keys()
    with open(filename, "w") as scores_file:
        scores_file.write(str(len(variables)))
        for child, dkt in local_scores.items():
            scores_file.write("\n" + child + " " + str(len(dkt.keys())))
            for parents, score in dkt.items():
                #highest_sup = None
                scores_file.write("\n" + str(score) + " " + str(len(parents)) +" "+ " ".join(parents))

def prune_local_score(this_score, parent_set, child_dkt):
    for other_parent_set, other_parent_set_score in child_dkt.items():
        if other_parent_set_score >= this_score and other_parent_set < parent_set:
            return True
    return False

def fromdataframe(df):
    cols = []
    arities = []
    varnames = []
    for varname, vals in df.items():
        varnames.append(varname)
        cols.append(vals.cat.codes)
        arities.append(len(vals.cat.categories))
    return np.transpose(np.array(cols,dtype=np.uint32)), arities, varnames

class Data:
    """
    Complete data (either discrete or continuous)

    This is an abstract class
    """

    def rawdata(self):
        '''
        The data without any information about variable names.

        Returns:
         numpy.ndarray: The data
        '''
        return self._data

    def variables(self):
        '''
        Returns:
         list : The variable names
        '''
        return self._variables

    def varidx(self):
        '''
        Returns:
         dict : Maps a variable name to its position in the list of variable names.
        '''
        return self._varidx

    
class DiscreteData(Data):
    """
    Complete discrete data
    """

    _value_type = np.uint8
    _arity_type = np.uint8
    _count_type = np.uint32
    
    def __init__(self, data_source, varnames = None, arities = None):
        '''Initialises a `DiscreteData` object.

        If  `data_source` is a filename then it is assumed that:

            #. All values are separated by whitespace
            #. Comment lines start with a '#'
            #. The first line is a header line stating the names of the 
               variables
            #. The second line states the arities of the variables
            #. All other lines contain the actual data

        Args:
          data_source (str/array_like/Pandas.DataFrame) : 
            Either a filename containing the data or an array_like object or
            Pandas data frame containing it.

          varnames (iter) : 
           Variable names corresponding to columns in the data.
           Ignored if `data_source` is a filename or Pandas DataFrame (since they 
           will supply the variable names). Otherwise if not supplied (`=None`)
           then variables names will be: X1, X2, ...

          arities (iter) : 
           Arities for the variables corresponding to columns in the data.
           Ignored if `data_source` is a filename or Pandas DataFrame (since they 
           will supply the arities). Otherwise if not supplied (`=None`)
           the arity for each variable will be set to the number of distinct values
           observed for that variable in the data.
        '''

        if type(data_source) == str:
            with open(data_source, "r") as file:
                line = file.readline().rstrip()
                while line[0] == '#':
                    line = file.readline().rstrip()
                varnames = line.split()
                line = file.readline().rstrip()
                while line[0] == '#':
                    line = file.readline().rstrip()
                arities = np.array([int(x) for x in line.split()],dtype=self._arity_type)

                for arity in arities:
                    if arity < 2:
                        raise ValueError("This line: '{0}' is interpreted as giving variable arities but the value {1} is less than 2.".format(line,arity))

                # class whose instances are callable functions 'with memory'
                class Convert:
                    def __init__(self):
                        self._last = 0 
                        self._dkt = {}

                    def __call__(self,s):
                        try:
                            return self._dkt[s]
                        except KeyError:
                            self._dkt[s] = self._last
                            self._last += 1
                            return self._dkt[s]


                converter_dkt = {}
                for i in range(len(varnames)):
                    # trick to create a function 'with memory'
                    converter_dkt[i] = Convert()
                data = np.loadtxt(file,
                                  dtype=self._value_type,
                                  converters=converter_dkt,
                                  comments='#')

        elif type(data_source) == pd.DataFrame:
            data, arities, varnames = fromdataframe(data_source)
        else:
            data = np.array(data_source,dtype=self._value_type)
        self._data = data
        if arities is None:
            self._arities = np.array([x+1 for x in data.max(axis=0)],dtype=self._arity_type)
        else:
            self._arities = np.array(arities,dtype=self._arity_type)

        # ensure _variables is immutable _varidx is always correct.
        if varnames is None:
            self._variables = tuple(['X{0}'.format(i) for i in range(1,len(self._arities)+1)])
        else:
            # order of varnames determined by header line in file, if file used
            self._variables = tuple(varnames)

        self._unique_data, counts = np.unique(self._data, axis=0, return_counts=True)
        self._unique_data_counts = np.array(counts,self._count_type)
            
        self._maxflatcontabsize = 1000000

        self._varidx = {}
        for i, v in enumerate(self._variables):
            self._varidx[v] = i
        self._data_length = data.shape[0]

        # create AD tree, if possible
        if adtree_available:
            #dd = np.transpose(np.vstack((self._arities,self._data))).astype(np.uint8,casting='safe')
            #print(dd)
            self._adtree = return_adtree(2000,1000,1000,np.transpose(np.vstack((self._arities,self._data))).astype(np.uint8,casting='safe'))
        
        
    def data(self):
        '''
        The data with all values converted to unsigned integers.

        Returns:
         pandas.DataFrame: The data
        '''

        df = pd.DataFrame(self._data,columns=self._variables)
        arities = self._arities
        for i, (name, data) in enumerate(df.items()):
            # ensure correct categories are recorded even if not
            # all observed in data
            df[name] = pd.Categorical(data,categories=range(arities[i]))
        return df
    
    def data_length(self):
        '''
        Returns:
         int: The number of datapoints in the data
        '''
        return self._data_length

    def arities(self):
        '''
        Returns:
         numpy.ndarray: The arities of the variables.
        '''
        return self._arities

    def arity(self,v):
        '''
        Args:
         v (str) : A variable name
        Returns:
         int : The arity of `v`
        '''

        return self._arities[self._varidx[v]]


    def contab(self,variables):
        cols = np.array([self._varidx[v] for v in variables], dtype=np.uint32)
        cols.sort() 
        return make_contab(self._unique_data,self._unique_data_counts,cols,self._arities[cols],self._maxflatcontabsize)[0]

    def make_contab_adtree(self,variables):
        '''
        Compute a marginal contingency table from data or report
        that the desired contingency table would be too big.
        
        Args:
         variables (iter): The variables in the marginal contingency table.


        Returns:
         tuple: 1st element is of type ndarray: 
          If the contingency table would have too many then the array is empty
          (and the 2nd element of the tuple should be ignored)
          else an array of counts of length equal to the product of the `arities`.
          Counts are in lexicographic order of the joint instantiations of the columns (=variables)
          2nd element: the 'strides' for each column (=variable)
        '''
        cols = np.array([self._varidx[v] for v in variables], dtype=np.uintc)
        cols.sort()
        p = len(cols)
        idx = p-1
        stride = 1
        arities = self._arities[cols]
        maxsize = self._maxflatcontabsize
        strides = np.empty(p,dtype=np.uint32)
        while idx > -1:
            strides[idx] = stride
            stride *= arities[idx]
            if stride > maxsize:
                return np.empty(0,dtype=np.uint32), strides
            idx -= 1
        #print(variables,cols,int(stride),flush=True)
        flatcontab = np.empty(stride,dtype=np.uintc)
        makecontab(self._adtree,cols,flatcontab)
        #print(flatcontab,flush=True)
        return flatcontab, strides
    
class ContinuousData(Data):
    """
    Complete continuous data
    """

    def __init__(self, data, varnames=None, header=True, comments='#', delimiter=None, standardise=False):
        '''Continuous data

        Args:
            data (numpy.ndarray/str) : The data (either as an array or a filename containing the data)
            varnames (iterable/None) : The names of the variables. If not given
             (=None) then if  `data` is a file having the variable names as a header then these are used
             else the variables are named X1, X2, X3, etc
            header (bool) : Ignored if `data` is not a filename. 
             Whether a header containing variable names is the first non-comment line in the file.
            comments (str) : Ignored if `data` is not a filename. Lines starting with this string are treated as comments.
            delimiter (None/str) : Ignored if `data` is not a filename. String used to separate values. If None then whitespace is used. 
            standardise (bool) : Whether to standardise the date to have mean 0 and sd = 1.
        '''
        if type(data) == str:
            skiprows = 0
            if header:
                with open(data) as f:            
                    for line in f:
                        if line.startswith(comments):
                             skiprows +=1
                        else:
                            vs = line.strip().split(sep=delimiter)
                            if varnames is None:  # only overwrite varnames if none given 
                                varnames = vs
                            skiprows += 1
                            break
            data = np.loadtxt(data, dtype = float, comments = comments, delimiter=delimiter, skiprows = skiprows)
        else:
            if type(data) == pd.DataFrame:
                varnames = list(data.columns)
            data = np.array(data, dtype=float)

        try:
            n, p = data.shape
        except ValueError:
            raise ValueError("Data must be a 2-d array")

        if standardise:
            self._data = (data - np.mean(data,axis=0)) / np.std(data,axis=0)
        else:
            self._data = data

        
        if varnames is None:
            varnames = ['X{0}'.format(i+1) for i in range(p)]
        if len(varnames) != p:
            raise ValueError("Expected {0} variable names, got {1}".format(p,len(varnames)))
    
        # store everything
        self._n = n
        self._p = p
        self._variables = tuple(varnames)
        self._cache = {}
        
        self._varidx = {}
        for i, v in enumerate(varnames):
            self._varidx[v] = i
        self._data_length = data.shape[0]


    def data(self):
        '''
        The data as a Pandas dataframe.

        Returns:
         pandas.DataFrame: The data
        '''

        return pd.DataFrame(self._data,columns=self._variables)

# Changed
# Add the class `MixedData` which deals with mixed data
class MixedData(Data):
    """
    Complete mixed data
    """
    
    _discrete_value_type = np.uint8
    _continuous_value_type = np.float
    _arity_type = np.uint8
    _count_type = np.uint32
    _index_type = np.uint8
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'
    
    def __init__(self, data_source, varnames=None, arities=None):
        '''Initializes a MixedData object.
        
        If `data_source` is a filename then it is assumed that:

            #. All values are separated by whitespace
            #. Comment lines start with a '#'
            #. The first line is a header line stating the names of the 
               variables
            #. The second line states the arities of the variables
            #. All other lines contain the actual data

        Args:
            data_source (str/array_like/Pandas.DataFrame) : 
                Either a filename containing the data or a tuple of two numpy.ndarrays or
                Pandas data frame containing it.

            varnames (iter) : 
                Variable names corresponding to columns in the data.
                Ignored if `data_source` is a filename or Pandas DataFrame (since they
                will supply the variable names). Otherwise if not supplied (`=None`)
                then discrete variable names will be: D1, D2, ..., and
                continuous variable names will be: C1, C2, ...
            arities (iter) : 
                Arities for the variables corresponding to columns in the data.
                Ignored if `data_source` is a filename or Pandas DataFrame (since they 
                will supply the arities). Otherwise if not supplied (`=None`)
                the arity for each variable will be set to the number of distinct values
                observed for that variable in the data.
        '''
        
        if type(data_source) == str:
            with open(data_source, 'r') as file:
                ## get variable names
                line = file.readline().rstrip()
                while line[0] == '#':
                    line = file.readline().rstrip()
                varnames = line.split()
                
                ## get arities
                line = file.readline().rstrip()
                while line[0] == '#':
                    line = file.readline().rstrip()
                arities = []
                varidx = dict()
                dis_idx = []
                dvnames = []
                con_idx = []
                cvnames = []
                for i, x in enumerate(line.split()):
                    if x != '-':
                        if int(x) < 2:
                            raise ValueError("This line: '{0}' is interpreted as giving variable arities but the value {1} is less than 2.".format(line, int(x)))
                        arities.append(int(x))
                        varidx[varnames[i]] = (self.DISCRETE, len(dis_idx))
                        dis_idx.append(i)
                        dvnames.append(varnames[i])
                    else:
                        varidx[varnames[i]] = (self.CONTINUOUS, len(con_idx))
                        con_idx.append(i)
                        cvnames.append(varnames[i])
                
                data = np.loadtxt(file, dtype=np.str, comments='#')
                data_length = data.shape[0]
                continuous_data = data[:, con_idx].astype(self._continuous_value_type)
                discrete_data = data[:, dis_idx]
                for j in range(discrete_data.shape[1]):
                    dkt = dict()
                    last = 0
                    for i in range(discrete_data.shape[0]):
                        if discrete_data[i][j] in dkt.keys():
                            discrete_data[i][j] = dkt[discrete_data[i][j]]
                        else:
                            dkt[discrete_data[i][j]] = str(last)
                            discrete_data[i][j] = str(last)
                            last += 1
                discrete_data = discrete_data.astype(self._discrete_value_type)
                
        elif type(data_source) == pd.DataFrame:
            data_length = data_source.shape[0]
            dvnames = []
            cvnames = []
            arities = []
            discrete_data = []
            continuous_data = []
            varidx = dict()
            
            for name, vals in data_source.items():
                if vals.dtype == float:
                    cvnames.append(name)
                    varidx[name] = (self.CONTINUOUS, len(continuous_data))
                    continuous_data.append(list(vals))
                else:
                    dvnames.append(name)
                    varidx[name] = (self.DISCRETE, len(discrete_data))
                    arities.append(np.unique(vals).size)
                    cols = list(vals)
                    last = 0
                    dkt = dict()
                    for i in range(len(cols)):
                        if cols[i] in dkt.keys():
                            cols[i] = dkt[cols[i]]
                        else:
                            dkt[cols[i]] = last
                            cols[i] = last
                            last += 1
                    discrete_data.append(cols)
                    
            discrete_data = np.array(discrete_data, dtype=self._discrete_value_type).T
            continuous_data = np.array(continuous_data, dtype=self._continuous_value_type).T
            
        else:
            discrete_data = np.array([])
            continuous_data = np.array([])
            varidx = dict()
            for vals in data_source:
                if vals.dtype == float:
                    continuous_data = vals
                else:
                    discrete_data = vals
                    for j in range(vals.shape[1]):
                        last = 0
                        dkt = dict()
                        for i in range(vals.shape[0]):
                            if vals[i][j] in dkt.keys():
                                discrete_data[i][j] = dkt[vals[i][j]]
                            else:
                                dkt[vals[i][j]] = last
                                last += 1
                                discrete_data[i][j] = dkt[vals[i][j]]
                    
            if discrete_data.shape[0] != continuous_data.shape[0]:
                raise ValueError('The length of data is not equal: discrete_data={0}, continuous_data={1}'.format(discrete_data.shape, continuous_data.shape))
            else:
                discrete_data = discrete_data.astype(self._discrete_value_type)
                continuous_data = continuous_data.astype(self._continuous_value_type)
                data_length = discrete_data.shape[0]
                
            if varnames is None:
                dvnames = ['D{0}'.format(i) for i in range(1, discrete_data.shape[1] + 1)]
                cvnames = ['C{0}'.format(i) for i in range(1, continuous_data.shape[1] + 1)]
            elif len(varnames[0]) == discrete_data.shape[1] and len(varnames[1]) == continuous_data.shape[1]:
                dvnames = varnames[0]
                cvnames = varnames[1]
            elif len(varnames[1]) == discrete_data.shape[1] and len(varnames[0]) == continuous_data.shape[1]:
                dvnames = varnames[1]
                cvnames = varnames[0]
            else:
                raise ValueError("The number of variables in continuous data ({0}) or discrete data ({1}) does not match the number of given variable names ({2}, {3})".format(continuous_data.shape[1], discrete_data.shape[1], len(varnames[0]), len(varnames[1])))
                
        self._discrete_data = discrete_data
        self._continuous_data = continuous_data
        self._data_length = data_length
        self._discrete_variables = tuple(dvnames)
        self._continuous_variables = tuple(cvnames)
        if arities is None:
            self._arities = np.array([x+1 for x in discrete_data.max(axis=0)], dtype=self._arity_type)
        elif len(arities) == self._discrete_data.shape[1]:
            self._arities = np.array(arities, dtype=self._arity_type)
        else:
            raise ValueError("The list of arities {0} does not match the data shape {1}.".format(arities, self._discrete_data.shape))
            
        self._unique_data, indices, counts = np.unique(self._discrete_data, axis=0, return_counts=True, return_inverse=True)
        self._unique_data_indices = np.array(indices, self._index_type)
        self._unique_data_counts = np.array(counts, self._count_type)
        
        self._maxflatcontabsize = 1000000
            
        if len(varidx.keys()) == len(self._discrete_variables) + len(self._continuous_variables):
            self._varidx = varidx
        else:
            self._varidx = dict()
            for i, v in enumerate(self._discrete_variables):
                self._varidx[v] = (self.DISCRETE, i)
            for i, v in enumerate(self._continuous_variables):
                self._varidx[v] = (self.CONTINUOUS, i)
            
    def rawdata(self):
        '''
        The data without any information about variable names.
        
        Returns:
            numpy.ndarray, numpy.ndarray: The discrete data, the continuous data
        '''
        return self._discrete_data, self._continuous_data
    
    def variables(self):
        '''
        Returns:
            list: The variable names
        '''
        return self._discrete_variables + self._continuous_variables
    
    def data(self):
        '''
        The data with all values.
        
        Returns:
            pandas.DataFrame: The dataframe
        '''
        
        discrete_df = pd.DataFrame(self._discrete_data, columns=self._discrete_variables)
        print(discrete_df)
        for i, (name, data) in enumerate(discrete_df.items()):
            ## ensure correct categories are recorded even if not
            ## all observed in data
            discrete_df[name] = pd.Categorical(data, categories=range(self._arities[i]))
        continuous_df = pd.DataFrame(self._continuous_data, columns=self._continuous_variables)
        return pd.concat([discrete_df, continuous_df], axis=1)
    
    def data_length(self):
        '''
        Returns:
            int: The number of datapoints in the data
        '''
        return self._data_length
    
    def arities(self):
        '''
        Returns:
            numpy.ndarray: The arities of the discrete variables.
        '''
        return self._arities
    
    def arity(self, v):
        '''
        Args:
            v (str): A variable name
        Returns:
            int: The arity of v
        '''
        if self._varidx[v][0] == self.CONTINUOUS:
            raise ValueError('The variable {} is continuous, not discrete.'.format(v))
            
        return self._arities[self._varidx[v][1]]

# START Classes for penalised log-likelihood 

# Changed
# Make the class `_AbsLLPenalised` possible to deal with mixed data
class _AbsLLPenalised:
    '''Abstract class for discrete penalised log likelihood scores
    '''

    def __init__(self,data):
        '''Initialises a `_AbsLLPenalised` object.

        Args:
         data (DiscreteData/Continuous): data
        '''
        self.__dict__.update(data.__dict__)
        self._maxllh = {}
        if type(data) == ContinuousData:
            self._cov = np.cov(self._data,rowvar=False,bias=True)
            self._gaussianll_cache = {}
            self._log2pi1 = log(2*pi) + 1
        elif type(data) == DiscreteData:
            self._entropy_cache = {}
        elif type(data) == MixedData:
            self._cgll_cache = {}
            self._df_cache = {}
            self._logn = log(self._data_length)
            self._log2pi1 = log(2*pi) + 1
            
        if type(data) == MixedData:
            variables = self._discrete_variables + self._continuous_variables
        else:
            variables = self._variables
        for i, v in enumerate(variables):
            self._maxllh[v] = self.ll_score(v, variables[:i]+variables[i+1:])[0]

    
class AbsDiscreteLLScore(DiscreteData):
    '''Abstract class for discrete log likelihood scores
    '''

    def score(self,child,parents):
        '''
        Return LL score minus complexity penalty for `child` having `parents`, 
        and also upper bound on the score for proper supersets.

        To compute the penalty the number of joint instantations is multiplied by the arity
        of the child minus one. This value is then multiplied by log(N)/2 for BIC and 1 for AIC.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Raises:
         ValueError: If the number of joint instantations of the parents would cause an overflow
          when computing the penalty
        
        Returns:
         tuple: The local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        this_ll_score, numinsts = self.ll_score(child,parents)
        if numinsts is None:
            raise ValueError('Too many joint instantiations of parents {0} to compute penalty'.format(parents))
        penalty = numinsts * self._child_penalties[child]
        # number of parent insts will at least double if any added
        return this_ll_score - penalty, self._maxllh[child] - (penalty*2)

    def entropy(self,variables):
        '''
        Compute the entropy for the empirical distribution of some variables

        Args:
         variables (iter): Variables

        Returns:
         The entropy for the empirical distribution of `variables` and the number of joint
         instantiations of `variables` if not too big else None
        '''
        vset = frozenset(variables)
        try:
            return self._entropy_cache[vset]
        except KeyError:
            cols = np.array(sorted([self._varidx[x] for x in variables]), dtype=np.uint32)
            if adtree_available: # using global variable makes testing easier
                contab, strides = self.make_contab_adtree(variables)
            else:
                contab, strides = make_contab(self._unique_data, self._unique_data_counts, cols,
                                          self._arities[cols], self._maxflatcontabsize)
            # do it anyway!
            #contab, strides = make_contab(self._unique_data, self._unique_data_counts, cols,
            #                              self._arities[cols], self._maxflatcontabsize)
            #print(contab)
            
            numinsts = len(contab)
            if numinsts == 0:
                numinsts = None
                # need to resort to slower method, will move this to numba at some point
                uniqs, uniq_idxs = np.unique(self._unique_data[:,cols],axis=0,return_inverse=True)
                contab = np.zeros(len(uniqs),dtype=self._count_type)
                unique_data_counts = self._unique_data_counts
                for i in range(len(self._unique_data)):
                    contab[uniq_idxs[i]] += unique_data_counts[i]

            h = entropy(contab)
            self._entropy_cache[vset] = h, numinsts
            return h, numinsts

        
    def ll_score(self,child,parents):
        '''
        The fitted log-likelihood score for `child` having `parents`

        In addition to the score the number of joint instantations of the parents is returned.
        If this number would cause an overflow `None` is returned instead of the number.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: 
            (1) The fitted log-likelihood local score for the given family and 
            (2) the number of joint instantations of the parents (or None if too big)
        '''
        pah, numinsts = self.entropy(parents)
        return self._data_length * (pah - self.entropy(list(parents)+[child])[0]), numinsts

class DiscreteLL(AbsDiscreteLLScore):

    def __init__(self,data):
        '''Initialises a `DiscreteLL` object.

        Args:
         data (DiscreteData): data
        '''
        _AbsLLPenalised.__init__(self,data)



    def score(self,child,parents):
        '''
        Return the fitted log-likelihood score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The fitted log-likelihood local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        return self.ll_score(child,parents)[0], self._maxllh[child]

class DiscreteBIC(AbsDiscreteLLScore):

    def __init__(self,data,k=1):
        '''Initialises a `DiscreteBIC` object.

        Args:
         data (DiscreteData): data
         k (float): Multiply standard BIC penalty by this amount, so increase for sparser networks
        '''
        _AbsLLPenalised.__init__(self,data)
        fn = 0.5 * log(self._data_length)  # Carvalho notation
        self._child_penalties = {v:k*fn*(self.arity(v)-1) for v in self._variables}


class DiscreteAIC(AbsDiscreteLLScore):

    def __init__(self,data,k=1):
        '''Initialises an `DiscreteAIC` object.

        Args:
         data (DiscreteData): data
         k (float): Multiply standard AIC penalty by this amount, so increase for sparser networks
        '''
        _AbsLLPenalised.__init__(self,data)
        self._child_penalties = {v:k*(self.arity(v)-1) for v in self._variables}

class AbsGaussianLLScore(ContinuousData):
    """
    Abstract class for Gaussian log-likelihood scoring
    """


    def gaussianll(self,variables):
        '''
        Compute the Gaussian log-likelihood of some variables

        Args:
         variables (iter): Variables

        Returns:
         The Gaussian log-likelihood of some variables
        '''
        vset = frozenset(variables)
        try:
            raise KeyError()
            return self._gaussianll_cache[vset]
        except KeyError:
            indices = [self._varidx[v] for v in variables]
            subcov = self._cov[np.ix_(indices,indices)]
            gll = -0.5 * self._data_length * (np.linalg.slogdet(subcov)[1] + len(indices)*self._log2pi1)
            self._gaussianll_cache[vset] = gll
            return gll
    
    def ll_score(self,child,parents):
        '''The Gaussian log-likelhood score for a given family, plus a dummy upper bound

        Args:
         child (str): The child variable
         parents (iter) : The parents
       
        Returns:
         tuple: (1) float: the Gaussian log-likelihood score for the family for current data
         (2) int: the number of parameters which is number of parents plus 2 (intercept plus sd of residuals) (int)
        '''
        return (self.gaussianll(list(parents)+[child]) - self.gaussianll(parents)), len(parents)+2

class GaussianLL(AbsGaussianLLScore):
    
    def __init__(self,data):
        '''Initialises an `GaussianLL` object.

        Args:
         data (ContinuousData): data
        '''
        _AbsLLPenalised.__init__(self,data)

    def score(self,child,parents):
        '''
        Return the fitted log-likelihood score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The fitted log-likelihood local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        return self.ll_score(child,parents)[0], self._maxllh[child]
    
    def score_dag(self, dag, by_node=False):
        '''
        Return the Gaussian score or local Gaussian scores for each node given a DAG based on the dataset.

        Args:
            dag (str): a model string DAG (e.g. `[A][B][C][D|F:E:G][F][E|A:B][G|A:F:E][H|A:D:B]`)
            by_node (bool): if it is false (default), this returns the Gaussian score of the given DAG.
                            If it is true, this returns the local Gaussian scores for each node.

        Returns:
            float or dict: The Gaussian score of the given DAG, or the local Gaussian scores for each node.
        '''
        families = {}
        for f in dag[1:-1].split(']['):
            family = f.split('|')
            if len(family) == 1:
                families[family[0]] = []
            elif len(family) == 2:
                families[family[0]] = family[1].split(':')
            else:
                raise ValueError('the given DAG is invalid: {}'.format(dag))

        scores = {}
        for v in self._variables:
            scores[v] = self.score(child=v, parents=families[v])

        if by_node:
            return scores
        else:
            total_ll = 0.
            total_maxlh = 0.
            for ss in scores.values():
                total_ll += ss[0]
                total_maxlh += ss[1]
            return total_ll, total_maxlh
        
class GaussianBIC(AbsGaussianLLScore):

    def __init__(self,data,k=1):
        '''Initialises an `GaussianBIC` object.

        Args:
         data (ContinuousData): data
         k (float): Multiply standard BIC penalty by this amount, so increase for sparser networks
        '''
        _AbsLLPenalised.__init__(self,data)
        # compute and store the sample covariance matrix (the version that gives the MLE)
        self._fn = k * 0.5 * log(self._data_length)  # Carvalho notation
        
    def score(self,child,parents):
        '''
        Return the fitted Gaussian BIC score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The Gaussian BIC local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        this_ll_score, numparams = self.ll_score(child,parents)
        return this_ll_score - self._fn * numparams, self._maxllh[child] - self._fn * (numparams+1)

class GaussianAIC(AbsGaussianLLScore):

    def __init__(self,data,k=1):
        '''Initialises an `GaussianAIC` object.

        Args:
         data (ContinuousData): data
         k (float): Multiply standard AIC penalty by this amount, so increase for sparser networks
        '''
        _AbsLLPenalised.__init__(self,data)
        self._k = k


    def score(self,child,parents):
        '''
        Return the fitted Gaussian AIC score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The Gaussian AIC local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        this_ll_score, numparams = self.ll_score(child,parents)
        return this_ll_score - self._k * numparams, self._maxllh[child] - self._k * (numparams+1)

class GaussianL0(AbsGaussianLLScore):
    '''
    Implements score discussed in "l_0-Penalized Maximum Likelihood for Sparse Directed
    Acyclic Graphs" by Sara van de Geer and Peter Buehlmann. Annals of Statistics 41(2):536-567, 2013.
    '''
    def __init__(self,data,k=1):
        '''Initialises an `GaussianL0` object.

        Args:
         data (ContinuousData): data
         k (float): Tuning parameter for L0 penalty. Called "lambda^2" in van de Geer and Buehlmann
        '''
        _AbsLLPenalised.__init__(self,data)
        self._k = k

    def score(self,child,parents):
        '''
        Return the fitted Gaussian AIC score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The Gaussian AIC local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        this_ll_score, numparams = self.ll_score(child,parents)
        sb = numparams-2 # just count edges
        return this_ll_score - self._k * sb, self._maxllh[child] - self._k * (sb+1)

# Changed
# Add the classes
# `AbsCGaussianLLScore`, `CGaussianLL`, `CGaussianBIC`, `CGaussianAIC`
# which compute the conditional Gaussian score of a BN which has both discrete and continuous variables

class AbsCGaussianLLScore(MixedData):
    '''
    Abstract class for mixed log likelihood scores
    '''
    
    def ll_score(self, child, parents):
        '''
        Compute the conditional Gaussian log-likelihood score for a given family, plus the degrees of freedom
        
        Args:
            child (str): The child variable
            parents (iter): The parent set
            
        Returns:
            tuple: (1) float: the conditional Gaussian log-likelihood score for the family for current data, and
            (2) int: the degrees of freedom
        '''
        ## prohibit the family with dicrete child and mixed parents
        ## comment out after ensuring the score function is correct
        # num = 0
        # for v in parents:
        #     if self._varidx[v][0] == self.CONTINUOUS:
        #         num += 1
        # if self._varidx[child][0] == self.DISCRETE and num > 0:
        #     return -float('inf'), 0
        ## until here
        cgll_parents_plus_child, df_parents_plus_child = self.c_gaussianll(list(parents) + [child])
        cgll_parents, df_parents = self.c_gaussianll(parents)
        return cgll_parents_plus_child - cgll_parents, df_parents_plus_child - df_parents
        
    def c_gaussianll(self, variables):
        '''
        Compute the conditional Gaussian log-likelihood of some variables
        
        Args:
            variables (iter): The variable set
            
        Returns:
            tuple: (1) float: The conditional Gaussian log-likelihood of some variables
            (2) int: The degrees of freedom
        '''
        vset = frozenset(variables)
        try:
            return self._cgll_cache[vset], self._df_cache[vset]
        except KeyError:
            dindices = []
            cindices = []
            for v in variables:
                vtype, i = self._varidx[v]
                if vtype == self.DISCRETE:
                    dindices.append(i)
                else:
                    cindices.append(i)
            dindices = sorted(dindices)
            cindices = sorted(cindices)
                    
            cgll = 0.
            df = 0.
            if len(dindices) == 0:
                result = self.con_gaussianll(cindices)
                cgll += result[0]
                df += result[1]
            elif len(cindices) == 0:
                result = self.dis_gaussianll(dindices)
                cgll += result[0]
                df += result[1]
            else:
                numinsts = 1
                for idx in dindices:
                    numinsts *= self._arities[idx]
                configurations, idxs = np.unique(self._discrete_data[:, dindices], axis=0, return_inverse=True)
                for i in range(configurations.shape[0]):
                    record_indices = list(np.where(idxs == i)[0])
                    result = self.mix_gaussianll(cindices, record_indices)
                    cgll += result[0]
                    df += result[1]
                result = self.mix_gaussianll(cindices, [])
                df += float(result[1]) * float(numinsts - configurations.shape[0])
                    
            self._cgll_cache[vset] = cgll
            self._df_cache[vset] = int(df - 1)
            return cgll, int(df - 1)
                    
    def mix_gaussianll(self, cindices, rindices):
        '''
        Compute the log-likelihood from the data partitioned by discrete variables
        It is used when the family members include both discrete and continuous variables
        
        Args:
            cindices (iter): Indices of continuous variables
            rindices (iter): Indices of records in data
            
        Returns:
            tuple: (1) float: the log-likelihood given the partitioned data
            (2) int: the degrees of freedom
        '''
        if len(rindices) < 2:
            return 0, int(0.5 * len(cindices) * (len(cindices) + 1)) + 1
        
        subdata = self._continuous_data[np.ix_(rindices, cindices)]
        subcov = np.cov(subdata, rowvar=False, bias=True)
        
        if subcov.size == 1:
            if subcov == 0:
                return 0, int(0.5 * len(cindices) * (len(cindices) + 1)) + 1
            else:
                logdetSigma = log(float(subcov))
        else:
            sign, logdetSigma = np.linalg.slogdet(subcov)
            if logdetSigma == -float('inf'):
                return 0, int(0.5 * len(cindices) * (len(cindices) + 1)) + 1
            else:
                logdetSigma = sign * logdetSigma
                
        mgll = -0.5 * subdata.shape[0] * (logdetSigma + len(cindices) * self._log2pi1) + subdata.shape[0] * log(subdata.shape[0] / self._data_length)
        return mgll, int(0.5 * len(cindices) * (len(cindices) + 1)) + 1
        
    def con_gaussianll(self, indices):
        '''
        Compute the log-likelihood from whole data
        It is used when all the family members are continuous variables

        Args:
            indices (iter): indices of continuous variables

        Returns:
            tuple: (1) float: the log-likelihood
            (2) int: the degrees of freedom
        '''
        subcov = np.cov(self._continuous_data[:, indices], rowvar=False, bias=True)
        if subcov.size == 1:
            logdetSigma = log(float(subcov))
        else:
            sign, logdetSigma = np.linalg.slogdet(subcov)
            logdetSigma = sign * logdetSigma
        cgll = -0.5 * self._data_length * (logdetSigma + len(indices) * self._log2pi1)
        return cgll, int(0.5 * len(indices) * (len(indices) + 1)) + 1
        
    def dis_gaussianll(self, indices):
        '''
        Compute the log-likelihood from whole data
        It is used when all the family members are discrete variables
        
        Args:
            indices (iter): indices of discrete variables

        Returns:
            tuple: (1) float: the log-likelihood
            (2) int: the degrees of freedom
        '''
        numinsts = 1
        for idx in indices:
            numinsts *= self._arities[idx]
            
        uniq_insts, uniq_idxs = np.unique(self._unique_data[:, indices], axis=0, return_inverse=True)
        contab = np.zeros(uniq_insts.shape[0]).astype(int)
        for i, j in enumerate(uniq_idxs):
            contab[j] += self._unique_data_counts[i]
            
        return np.sum(contab * np.log(contab/contab.sum())), numinsts

class CGaussianLL(AbsCGaussianLLScore):
    
    def __init__(self, data):
        '''
        Initialize a `CGaussianLL` object.
        
        Args:
            data (MixedData): data
        '''
        _AbsLLPenalised.__init__(self, data)
        
    def score(self, child, parents):
        '''
        Return the fitted log-likelihood score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.
        
        Args:
            child (str): The child variable
            parents (iter): The parent variables
        
        Returns:
            tuple: (1) float: The fitted log-likelihood local score for the given family
            (2) float: an upper bound on the local score for proper supersets of `parents`
        '''
        return self.ll_score(child, parents)[0], self._maxllh[child]

    def score_dag(self, dag, by_node=False):
        '''
        Return the CG score or local CG scores for each node given a DAG based on the dataset.

        Args:
            dag (str): a model string DAG (e.g. `[A][B][C][D|F:E:G][F][E|A:B][G|A:F:E][H|A:D:B]`)
            by_node (bool): if it is false (default), this returns the CG score of the given DAG.
                            If it is true, this returns the local CG scores for each node.

        Returns:
            float or dict: The CG score of the given DAG, or the local CG scores for each node.
        '''
        families = {}
        for f in dag[1:-1].split(']['):
            family = f.split('|')
            if len(family) == 1:
                families[family[0]] = []
            elif len(family) == 2:
                families[family[0]] = family[1].split(':')
            else:
                raise ValueError('the given DAG is invalid: {}'.format(dag))

        scores = {}
        for v in self._discrete_variables + self._continuous_variables:
            scores[v] = self.score(child=v, parents=families[v])

        if by_node:
            return scores
        else:
            total_ll = 0.
            for ss in scores.values():
                total_ll += ss[0]
            return total_ll

class CGaussianBIC(AbsCGaussianLLScore):

    def __init__(self,data,k=1):
        '''Initialises an `CGaussianBIC` object.

        Args:
         data (MixedData): data
         k (float): Multiply standard BIC penalty by this amount, so increase for sparser networks
        '''
        _AbsLLPenalised.__init__(self,data)
        # compute and store the sample covariance matrix (the version that gives the MLE)
        self._fn = k * 0.5 * log(self._data_length)  # Carvalho notation
        
    def score(self,child,parents):
        '''
        Return the fitted conditional Gaussian BIC score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
            child (str): The child variable
            parents (iter): The parent variables

        Returns:
         tuple: (1) float: The conditional Gaussian BIC local score for the given family and
         (2) float: an upper bound on the local score for proper supersets of `parents`
        '''
        this_ll_score, this_df = self.ll_score(child,parents)
        return this_ll_score - self._fn * this_df, self._maxllh[child] - self._fn * (this_df+1)

class CGaussianAIC(AbsCGaussianLLScore):

    def __init__(self,data,k=1):
        '''Initialises an `GaussianAIC` object.

        Args:
         data (ContinuousData): data
         k (float): Multiply standard AIC penalty by this amount, so increase for sparser networks
        '''
        _AbsLLPenalised.__init__(self,data)
        self._k = float(k)


    def score(self,child,parents):
        '''
        Return the fitted Gaussian AIC score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: (1) float: The Gaussian AIC local score for the given family and
         (2) float: an upper bound on the local score for proper supersets of `parents`
        '''
        this_ll_score, this_df = self.ll_score(child,parents)
        return this_ll_score - self._k * this_df, self._maxllh[child] - self._k * (this_df+1)

# END Classes for penalised log-likelihood 
    
class BDeu(DiscreteData):
    """
    Discrete data with attributes and methods for BDeu scoring
    """

    def __init__(self,data,alpha=1.0):
        '''Initialises a `BDeu` object.

        Args:
         data (DiscreteData): data
         
         alpha (float): The *equivalent sample size*
        '''
        self.__dict__.update(data.__dict__)
        self.alpha = alpha
        self._cache = {}

        # for upper bounds
        self._atoms = get_atoms(self._data,self._arities)

        
    @property
    def alpha(self):
        '''float: The *equivalent sample size* used for BDeu scoring'''
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        '''Set the *equivalent sample size* for BDeu scoring
        
        Args:
         alpha (float): the *equivalent sample size* for BDeu scoring

        Raises:
         ValueError: If `alpha` is not positive
        '''
        if not alpha > 0:
            raise ValueError('alpha (equivalent sample size) must be positive but was give {0}'.format(alpha))
        self._alpha = alpha

    def clear_cache():
        '''Empty the cache of stored BDeu component scores

        This should be called, for example, if new scores are being computed
        with a different alpha value
        '''
        self._cache = {}
        
    def upper_bound_james(self,child,parents,alpha=None):
        """
        Compute an upper bound on proper supersets of parents

        Args:
         child (str) : Child variable.
         parents (iter) : Parent variables
         alpha (float) : ESS value for BDeu score. If not supplied (=None)
          then the value of `self.alpha` is used.

        Returns:
         float : An upper bound on the local score for parent sets
         for `child` which are proper supersets of `parents`

        """
        if alpha is None:
            alpha = self._alpha
        child_idx = self._varidx[child]
        pa_idxs = sorted([self._varidx[v] for v in parents])
        for pa_idx in pa_idxs:
            alpha /= self._arities[pa_idx]
        r = self._arities[child_idx]

        # each element of atoms_ints is a tuple of ints:
        # (fullinst,childvalcounts,sum(childvalcounts),ok_first)
        # each element of atoms_floats is:
        # sum_n n*log(n/tot), where sum is over childvalcounts
        # and tot = sum(childvalcounts)
        atoms_ints, atoms_floats = self._atoms[0][child_idx], self._atoms[1][child_idx]

        if len(atoms_floats) == 0:
            return 0.0

        # remove cols corresponding to non-parents and order
        p = len(self._arities)
        end_idxs = list(range(p,p+r+2))
        atoms_ints_redux = atoms_ints[:,pa_idxs+end_idxs]
        if len(pa_idxs) > 0:
            idxs = np.lexsort([atoms_ints_redux[:,col] for col in range(len(pa_idxs))])
        else:
            idxs = list(range(len(atoms_floats)))

        return upper_bound_james_fun(atoms_ints_redux,atoms_floats,len(pa_idxs),alpha,r,idxs)
        
    def bdeu_score_component(self,variables,alpha=None):
        '''Compute the BDeu score component for a set of variables
        (from the current dataset).

        The BDeu score for a child v having parents Pa is 
        the BDeu score component for Pa subtracted from that for v+Pa
        
        Args:
         variables (iter) : The names of the variables
         alpha (float) : The effective sample size parameter for the BDeu score.
          If not supplied (=None)
          then the value of `self.alpha` is used.

        Returns:
         float : The BDeu score component.
        '''
        if alpha is None:
            alpha = self._alpha

        if len(variables) == 0:
            return lgamma(alpha) - lgamma(alpha + self._data_length), 1
        else:
            cols = np.array(sorted([self._varidx[x] for x in list(variables)]), dtype=np.uint32)
            return compute_bdeu_component(
                self._unique_data,self._unique_data_counts,cols,
                np.array([self._arities[i] for i in cols], dtype=self._arity_type),
                alpha,self._maxflatcontabsize)

    def _bdeu_score_component_cache(self,s):
        s_set = frozenset(s)
        try:
            score_non_zero_count = self._cache[s_set]
        except KeyError:
            score_non_zero_count = self.bdeu_score_component(s_set)
            self._cache[s_set] = score_non_zero_count
        return score_non_zero_count

    def bdeu_score(self, child, parents):

        parent_score, _ = self._bdeu_score_component_cache(parents)
        family_score, non_zero_count = self._bdeu_score_component_cache((child,)+parents)

        simple_ub = -log(self.arity(child)) * non_zero_count

        #james_ub = self.upper_bound_james(child,parents)
        
        #return parent_score - family_score, min(simple_ub,james_ub)
        return parent_score - family_score, simple_ub

        
    def bdeu_scores(self,palim=None,pruning=True,alpha=None):
        """
        Exhaustively compute all BDeu scores for all child variables and all parent sets up to size `palim`.
        If `pruning` delete those parent sets which have a subset with a better score.
        Return a dictionary dkt where dkt[child][parents] = bdeu_score
        
        Args:
         palim (int/None) : Limit on parent set size
         pruning (bool) : Whether to prune
         alpha (float) : ESS for BDeu score. 
          If not supplied (=None)
          then the value of `self.alpha` is used.



        Returns:
         dict : dkt where dkt[child][parents] = bdeu_score
        """
        if alpha is None:
            alpha = self._alpha
        
        if palim == None:
            palim = self._arities.size - 1

        score_dict = {}
        # Initialisation
        # Need to create dict for every child
        # also its better to do the zero size parent set calc here
        # so that we don't have to do a check for every parent set
        # to make sure it is not of size 0 when calculating score component size
        no_parents_score_component = lgamma(alpha) - lgamma(alpha + self._data_length)
        for c, child in enumerate(self._variables):
            score_dict[child] = {
                frozenset([]):
                no_parents_score_component
            }
        
        for pasize in range(1,palim+1):
            for family in combinations(self._variables,pasize): 
                score_component = self.bdeu_score_component(family,alpha)
                family_set = frozenset(family)
                for child in self._variables:
                    if child in family_set:
                        parent_set = family_set.difference([child])
                        score_dict[child][parent_set] -= score_component
                        if pruning and prune_local_score(score_dict[child][parent_set],parent_set,score_dict[child]):
                            del score_dict[child][parent_set]
                    else:
                        score_dict[child][family_set] = score_component 
                
        # seperate loop for maximally sized parent sets
        for vars in combinations(self._variables,palim+1):
            score_component = self.bdeu_score_component(vars,alpha)
            vars_set = frozenset(vars)
            for child in vars:
                parent_set = vars_set.difference([child])
                score_dict[child][parent_set] -= score_component
                if pruning and prune_local_score(score_dict[child][parent_set],parent_set,score_dict[child]):
                    del score_dict[child][parent_set]

            
        return score_dict

class BGe(ContinuousData):
    """
    Continuous data with attributes and methods for BGe scoring
    """
    def __init__(self, data, nu=None, alpha_mu = 1.0, alpha_omega = None, prior_matrix=None):
        '''Create a BGe scoring object

        Args:
            data (ContinuousData) : The data
            nu (numpy.ndarray/None) : the mean vector for the normal part of the 
             normal-Wishart prior. If not given (=None), then the sample mean
             is used.
            alpha_mu (float) : imaginary sample size for the normal part of the
              normal-Wishart prior. 
            alpha_omega (int/None) : The degrees of freedom for the Wishart 
             part of the normal-Wishart prior. Must exceed p-1 where p is the number of 
             variables. If not given (=None) then `alpha_omega` is set to p+2.
            prior_matrix (numpy.ndarray/None) : The prior matrix 'T'
             for the Wishart part of the normal-Wishart prior. If not given (=None), then
             this is set to t*I_n where t = alpha_mu*(alpha_omega-n-1)/(alpha_mu+1)
        '''
        
        self.__dict__.update(data.__dict__)
        p = self._p
        n = self._n
        data = self._data
        
        # No need to explicitly represent nu if it the sample mean vector
        if nu is not None and len(nu) != p:
            raise ValueError("nu is wrong length. Expected length is {0}, but got length of {1}".format(p,len(nu)))

        if not alpha_mu > 0:
                raise ValueError("alpha_mu must be positive, but is {0}".format(alpha_mu))
        
        if alpha_omega is None:
            alpha_omega = self._p + 2
        else:
            if type(alpha_omega) != int:
                raise ValueError("alpha_omega must be an integer but is {1}".format(alpha_omega))
            if not alpha_omega > p - 1:
                raise ValueError("alpha_omega must exceed p-1 = {0}, but is {1}".format(p-1,alpha_omega))

        if prior_matrix is None:
            explicit_prior_matrix = np.zeros((p,p))
            t = (alpha_mu * (alpha_omega - p - 1.0)) / (alpha_mu + 1.0)
            np.fill_diagonal(explicit_prior_matrix, t)
        else:
            explicit_prior_matrix = prior_matrix
            d1, d2 = explicit_prior_matrix.shape
            if d1 != p or d2 != p:
                raise ValueError("prior_matrix must be {0} X {0} but is {1} X {2}".format(p,d1,d2))
            # need to add check for prior_matrix being positive definite

        # for debugging ...
        #print(explicit_prior_matrix)
            
        # compute (log) prefactors for each possible size of parent set
        log_prefactors = []
        const_log_prefactor = 0.5 * (log(alpha_mu) - log(n + alpha_mu))
        for pasize in range(p):
            log_prefactor = (
                const_log_prefactor
                + gammaln(0.5*(n + alpha_omega - p + pasize + 1))
                - gammaln(0.5*(alpha_omega - p + pasize + 1))
                - (0.5*n) * log(pi))
            log_prefactors.append(log_prefactor)
        # If using 'default' prior matrix then include ratio of dets of prior matrices here:
        if prior_matrix is None:
            logt = log(t)
            const_term = 0.5 * (alpha_omega - p + 1) 
            for pasize in range(p):
                log_prefactors[pasize] += (const_term + pasize) * logt

        # compute posterior matrix 'R'
        # need rowvar=F since each row (not column) is a datapoint
        s_n = (n-1) * np.cov(data,rowvar=False)
        posterior_matrix = np.add(explicit_prior_matrix,s_n)
        if nu is not None:
            diff_vec = np.subtract(nu,np.mean(data,axis=0))
            # (4) in Kuipers et al is wrong, since it has alpha_omega
            # when it should be alpha_mu in (n*alpha_mu)/(n+alpha_mu)
            posterior_matrix = np.add(posterior_matrix,
                                      ((n*alpha_mu)/(n+alpha_mu))*np.outer(diff_vec,diff_vec))
        # for debugging ...
        #print(posterior_matrix)
            
        # store everything
        self._alpha_mu = alpha_mu
        self._alpha_omega = alpha_omega
        self._log_prefactors = log_prefactors
        self._prior_matrix = prior_matrix # NB could be None
        self._posterior_matrix = posterior_matrix
        self._cache = {}

    
    def bge_component(self,vs):
        '''Compute the BGe component for given variables

        The BGe score for a family child<-parents is the component for child+parents
        minus the component for parents (+ a constant term which just depends on the number
        of parents).

        Args:
         vs (iter): Variable names
        
        Returns:
         float: The BGe component for the given variables
        '''
        vset = frozenset(vs)
        try:
            component = self._cache[vset]
        except KeyError:
            # since we ultimately want absolute value of determinant, no need to worry about order
            # unless, perhaps, there a performance issue?
            indices = [self._varidx[v] for v in vs]
            # Get 'R_PP' or 'R_QQ'
            array_indices = np.ix_(indices,indices)
            posterior_matrix = self._posterior_matrix[array_indices]
            component = -(0.5 *
                         (self._n + self._alpha_omega - self._p + len(vs)) *
                         (np.linalg.slogdet(posterior_matrix)[1]))
            if self._prior_matrix is not None:
                # Get 'T_PP' or 'T_QQ'
                prior_matrix = self._prior_matrix[array_indices]
                component += (0.5 *
                              (self._alpha_omega - self._p + len(vs)) *
                              (np.linalg.slogdet(prior_matrix)[1]))
            self._cache[vset] = component
        return component
            
    def bge_score(self,child,parents):
        '''The BGe score for a given family, plus upper bound

        Args:
         child (str): The child variable
         parents (iter) : The parents
       
        Returns:
         tuple: First element of tuple isf the BGe score for the family for current data (using current hyperparameters)
          Second element is an upper bound.
        '''
        return (self._log_prefactors[len(parents)] +
                self.bge_component(list(parents)+[child]) -
                self.bge_component(parents), None)

