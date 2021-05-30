void* return_adtree(
   int rmin,            /**< if count below this then create a leaflist */
   int adtreedepthlim,  /**< limit on the depth of the ADTREE */
   int adtreenodeslim,  /**< limit on the number of nodes in the ADTREE */
   unsigned char* data,     /**< data[i*ndatapoints_plus_one+j+1] is the value of variable i in datapoint j, 
                               data[i*ndatapoints_plus_one] is arity of variable i */
   int nvars,               /**< Number of variables in the data */
   int ndatapoints_plus_one /**< Number of datapoints in the data plus one */
   );

void makecontab(
   void* v_adtree_etc,      /**< (Pointer to) the ADTREE */
   unsigned int* variables, /**< Variables in the sought contingency table (sorted) */
   int nvariables,          /**< Number of variables in the contingency table */
   unsigned int* flatcontab,
   int flatcontabsize
   );

void del_adtree(
   void *adtree,        /**< pointer to ADTREE being deleted */
   int nvars,           /**< Number of variables in the data */
   int *arity           /**< arity[i] is the arity of variable i, */
   );

