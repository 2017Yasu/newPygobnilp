%module adtree
%{
#define SWIG_FILE_WITH_INIT
#include "adtree.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (unsigned char* IN_ARRAY2, int DIM1, int DIM2) {(unsigned char* data, int nvars, int ndatapoints_plus_one )};
%apply (unsigned int* IN_ARRAY1, int DIM1) {(unsigned int* variables, int nvariables)};
%apply (unsigned int* INPLACE_ARRAY1, int DIM1) {(unsigned int* flatcontab, int flatcontabsize)};
%include "adtree.h"





