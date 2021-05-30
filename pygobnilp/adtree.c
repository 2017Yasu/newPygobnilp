#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include "adtree.h"

#define BLOCKSIZE 10000
#define MAXARITY UCHAR_MAX

typedef unsigned int VARIABLE;   /**< Variable in the data */
/* indexing with "int" is supposed to be quicker but empirically
   it has been shown that "unsigned char" is substantially faster,
   presumably due to memory savings */
typedef unsigned int COUNT;      /**< Count, typically of datapoints */
typedef double SCORE;            /**< A local score. Currently only BDeu, BGe and BIC implemented */

typedef unsigned int ROW;        /**< Index of a row (i.e.\ datapoint) in the data */
typedef unsigned char ARITY;     /**< Arity of a variable in the data */
typedef unsigned char VALUE;     /**< Value of a variable in the data */
typedef COUNT* FLATCONTAB;       /**< Flat contingency table */


/** represents data for a 'query', e.g.\ X1=1,X3=2
 *
 * There is always a count of how many datapoints satisfy the query.
 * If, as is most common,  the highest-indexed variable (in the entire dataset) is not mentioned in the query then:
 * if the count < rmin there is a pointer to the datapoint indices for the data
 * otherwise there is a pointer to an 'array' of 'vary nodes' one for each
 * of the remaining variables.
 */
struct adtree
{
   COUNT count;                /**< how many datapoints for this query */
   struct varynode *children;  /**< one for each variable specialising  the query, if any 
                                  (NULL if leaflist is used or there are no specialisations)*/
   ROW *leaflist;              /**< leaflist, if used (NULL otherwise ) */
};
typedef struct adtree ADTREE;  /**< An AD tree */



struct varynode                 /** for splitting data on values of a particular variable (variable not stored in varynode) */
{
   struct adtree **children;    /**< children[val] is a pointer to the ADTREE for the specialisation var=val 
                                   (or NULL if no data for this, mcv) */
   VALUE mcv;                   /**< most common value for this variable (in appropriate 'conditional' dataset) */
};
typedef struct varynode VARYNODE; /**< A varynode (in an AD tree) */


union treecontab                 /** A tree-shaped contingency table */
{
   union treecontab *children;   /**< when there are variables... 
                                  if treecontab.children == NULL then there are only zero counts in the contingency table.
                                  if treecontab.children != NULL then treecontab.children[i] is the treecontab formed by specialising on
                                  the ith value of the first variable */
   COUNT count;                  /**< when there are no variables treecontab.count just stores a count */
};
typedef union treecontab TREECONTAB; /**< A tree-shaped contingency table. The variables for the contingency table are not stored in
                                    this data structure. */

struct adtree_etc
{
   ADTREE* adtree;
   VALUE** data;
   ARITY* arity;
   VARIABLE nvars;
};
typedef struct adtree_etc ADTREE_ETC;  /**< An AD tree with data and arities*/


void build_varynode(VARYNODE *varynode, VARIABLE variable, ROW *theserows, COUNT count, int rmin, const int depth, int *n_nodes, const int adtreedepthlim, const int adtreenodeslim, VARIABLE nvars, VALUE **data, ARITY *arity);


/** Build an AD tree from (a subset of) the data */
void build_adtree(
   ADTREE *adtree,            /**< pointer to ADTREE being built */
   const VARIABLE variable,   /**< first variable to specialise further on, if variable=nvars then there is none */
   ROW *theserows,            /**< datapoint indices for this tree */
   COUNT count,               /**< number of datapoints for this tree */
   const int rmin,            /**< if count below this then create a leaflist */
   const int depth,           /**< the depth of this node */
   int *n_nodes,              /**< (pointer to) the number of nodes in the ADTREE */
   const int adtreedepthlim,  /**< limit on the depth of the ADTREE */
   const int adtreenodeslim,  /**< limit on the number of nodes in the ADTREE */
   const VARIABLE nvars,      /**< Number of variables in the data */
   VALUE **data,              /**< data[i][j] is the value of variable i in row j */
   ARITY *arity               /**< arity[i] is the arity of variable i, */
)
{
   COUNT j;
   VARIABLE var;

   assert(variable < nvars + 1);
   assert(count > 0);
   assert(theserows != NULL);
   assert(adtree != NULL);

   adtree->count = count;
   adtree->leaflist = NULL;
   adtree->children = NULL;

   /* if there can be no further splitting just record count */
   if( variable < nvars )
   {
      /* if count small enough then make a leaflist which is a copy of theserows */
      /* if depth too large or number of nodes too large, similarly just dump records in a leaflist */
      if( (int) count < rmin || depth > adtreedepthlim || *n_nodes > adtreenodeslim )
      {
         adtree->leaflist = (ROW *) malloc(count * sizeof(ROW));
         for( j = 0; j < count; ++j )
            adtree->leaflist[j] = theserows[j];
      }
      /* or create vary nodes - one for each further variable - and recurse */
      else
      {
         adtree->children = (VARYNODE *) malloc((nvars - variable) * sizeof(VARYNODE));
         if( adtree->children == NULL )
            printf("Couldn't allocate memory for vary nodes\n");
         for( var = variable; var < nvars; ++var )
            build_varynode((adtree->children) + (var - variable), var, theserows, count, rmin, depth, n_nodes, adtreedepthlim, adtreenodeslim, nvars, data, arity);
      }
   }

   /* can always free since data indices are always copied */
   free(theserows);
   return;
}
/** Build an vary node from (a subset of) the data */
void build_varynode(
   VARYNODE *varynode,        /**< varynode being built */
   VARIABLE variable,         /**< which variable is being split */
   ROW *theserows,            /**< datapoint indices to divide between values of variable */
   COUNT count,               /**< number of datapoints for this tree */
   const int rmin,            /**< if count below this then create a leaflist */
   int depth,                 /**< the depth of this node */
   int *n_nodes,              /**< (pointer to) the number of nodes in the ADTREE */
   const int adtreedepthlim,  /**< limit on the depth of the ADTREE */
   const int adtreenodeslim,  /**< limit on the number of nodes in the ADTREE */
   VARIABLE nvars,            /**< Number of variables in the data */
   VALUE **data,              /**< data[i][j] is the value of variable i in row j */
   ARITY *arity               /**< arity[i] is the arity of variable i, */
)
{

   const VALUE *thisdata = data[variable];
   const ARITY thisarity = arity[variable];
   ROW **childdata;
   COUNT *childcount;
   VALUE val;
   COUNT j;
   VALUE mcv = 0;
   COUNT countmcv = 0;
   ROW row;

   assert(variable < nvars);
   assert(varynode != NULL);
   assert(theserows != NULL);
   assert(count > 0);


   /* initialise data structures for splitting data on values of the variable */
   childdata = (ROW **) malloc(thisarity * sizeof(ROW *));
   if( childdata == NULL )
      printf("Couldn't allocate childdata\n");
   childcount = (COUNT *) malloc(thisarity * sizeof(COUNT));
   if( childcount == NULL )
      printf("Couldn't allocate childcount\n");

   for( val = 0; val < thisarity; ++val )
   {
      /* lazily allocate space of size 'count' for each val
         (which is certainly big enough), perhaps should
         do allocate in small blocks, on demand
      */
      childdata[val] = (ROW *) malloc(count * sizeof(ROW));
      if( childdata[val] == NULL )
         printf("Couldn't allocate childdata_val\n");

      childcount[val] = 0;
   }

   /* split the data for this tree on values of the variable */
   for( j = 0; j < count; ++j )
   {
      row = theserows[j];
      val = thisdata[row];
      childdata[val][childcount[val]++] = row;
   }


   /* find most common value */
   for( val = 0; val < thisarity; ++val )
      if( childcount[val] > countmcv )
      {
         countmcv = childcount[val];
         mcv = val;
      }
   assert(countmcv > 0);
   varynode->mcv = mcv;

   /* throw away rows for mcv and any zero counts resize the others */
   /* resize as soon as possible */
   for( val = 0; val < thisarity; ++val )
   {
      if( val == mcv || childcount[val] == 0 )
         free(childdata[val]);
      else
      {
         childdata[val] = (ROW *) realloc(childdata[val], childcount[val] * sizeof(ROW));
         if( childdata[val] == NULL )
            printf("Couldn't re-allocate childdata_val\n");
      }
   }

   varynode->children = (ADTREE **) malloc(thisarity * sizeof(ADTREE *));
   if( varynode->children == NULL )
      printf("Couldn't allocate memory for AD trees\n");

   variable++;          /* can lead to variable=nvars, ie a fake 'extra' variable */
   for( val = 0; val < thisarity; ++val )
   {
      if( val == mcv || childcount[val] == 0 )
         varynode->children[val] = NULL;
      else
      {
         /* childdata[val] freed in build_adtree (unless it becomes a leaflist) */
         varynode->children[val] = (ADTREE *) malloc(sizeof(ADTREE));
         (*n_nodes)++;
         if( varynode->children[val] == NULL )
            printf("Couldn't allocate memory for AD tree\n");

         build_adtree(varynode->children[val], variable, childdata[val], childcount[val],
            rmin, depth + 1, n_nodes, adtreedepthlim, adtreenodeslim, nvars, data, arity);
      }
   }

   free(childdata);
   free(childcount);

   return;

}

static void print_varynode(VARYNODE* varynode, const VARIABLE variable, VARIABLE nvars, ARITY *arity);

/** Print an AD tree (for debugging) */
static
void print_adtree(
   ADTREE *adtree,          /**< pointer to ADTREE being deleted */
   const VARIABLE variable, /**< first variable to specialise further on, if variable=nvars then there is none */
   VARIABLE nvars,          /**< Number of variables in the data */
   ARITY *arity             /**< arity[i] is the arity of variable i, */
   )
{
   printf("adtree=%p,count=%d,firstvarspec=%d,leaflist=%p,children=%p\n",
      (void*)adtree,adtree->count,variable,(void*)adtree->leaflist,(void*)adtree->children);
   if( adtree->children != NULL )
   {
      VARIABLE var;
      printf("%d children:\n",nvars-variable);
      for( var = variable; var < nvars; ++var )
      {
         printf("varynode %p for variable %d\n", (void*)((adtree->children) + (var-variable)), var);
      }
      printf("\n");
      for( var = variable; var < nvars; ++var )
      {
         /* printf("varynode for variable %d\n",var); */
         print_varynode((adtree->children) + (var - variable),var,nvars,arity);
      }
   }
   if( adtree->leaflist != NULL )
   {
      printf("leaflist of size %d\n",adtree->count);
      /* COUNT j; */
      /* printf("leaflist="); */
      /* for( j = 0; j < adtree->count; ++j ) */
      /*    printf("%d,",adtree->leaflist[j]); */
      /* printf("END\n"); */
   }
  
}

/** Print an AD tree (for debugging) */
static
void print_varynode(
   VARYNODE* varynode,      /**< varynode to print */
   const VARIABLE variable, /**< which variable is being split */
   VARIABLE nvars,          /**< Number of variables in the data */
   ARITY *arity             /**< arity[i] is the arity of variable i, */
   )
{
   const ARITY thisarity = arity[variable];
   VALUE val;
   
   printf("varynode=%p,firstvarspec=%d,arity=%d,children=%p,mcv=%d\n",
      (void*)varynode,variable,thisarity,(void*)varynode->children,varynode->mcv);
   printf("%d children:\n",thisarity);
   for( val = 0; val < thisarity; ++val )
   {
      printf("adtree %p for value %d\n", (void*)varynode->children[val], val);
   }
   printf("\n");
   for( val = 0; val < thisarity; ++val )
   {
      printf("val=%d\n",val);
      if( varynode->children[val] == NULL )
         printf("NULL\n");
      else
         print_adtree(varynode->children[val],variable+1,nvars,arity);
   }
   printf("\n");
}


ADTREE* ret_adtree(
   const int rmin,            /**< if count below this then create a leaflist */
   const int adtreedepthlim,  /**< limit on the depth of the ADTREE */
   const int adtreenodeslim,  /**< limit on the number of nodes in the ADTREE */
   const VARIABLE nvars,      /**< Number of variables in the data */
   const COUNT nrows,         /**< Number of datapoints in the data */
   VALUE **data,              /**< data[i][j] is the value of variable i in row j */
   ARITY *arity               /**< arity[i] is the arity of variable i, */
   )
{

   ADTREE* adtree = malloc(sizeof(ADTREE)); 
   ROW* allrows; 
   int n_nodes = 0;
   int i;
   
   allrows = (ROW *) malloc(nrows * sizeof(ROW)); 
   for( i = 0; i < nrows; ++i ) 
      allrows[i] = i; 
   
   build_adtree(adtree, 0, allrows, nrows, rmin, 0, &n_nodes,
      adtreedepthlim, adtreenodeslim, nvars, data, arity);

   /* printf("Just made adtree.\n"); */
   /* print_adtree(adtree,0,nvars,arity); */

   return adtree;
}

void* return_adtree(
   int rmin,                /**< if count below this then create a leaflist */
   int adtreedepthlim,      /**< limit on the depth of the ADTREE */
   int adtreenodeslim,      /**< limit on the number of nodes in the ADTREE */
   unsigned char* data,     /**< data[i*ndatapoints_plus_one+j+1] is the value of variable i in datapoint j, 
                               data[i*ndatapoints_plus_one] is arity of variable i */
   int nvars,               /**< Number of variables in the data */
   int ndatapoints_plus_one /**< Number of datapoints in the data plus one */
   )
{
   ADTREE* adtree;
   VALUE* mydata;
   VALUE** data2;
   int i;
   int j;
   int k;
   int l;
   ADTREE_ETC* adtree_etc = malloc(sizeof(ADTREE_ETC)); 
   ARITY* arities;
   int ndatapoints = ndatapoints_plus_one - 1;
   int data_length;

   data_length = nvars * ndatapoints;
   
   mydata = (VALUE *) malloc(data_length * sizeof(VALUE));
   /* need data2[i][j] to be value of variable i for datapoint j */
   data2 = (VALUE **) malloc(nvars * sizeof(VALUE *));
   arities = (ARITY *) malloc(nvars * sizeof(ARITY *));
   j = 0; /* index for my data */
   k = 0; /* index for incoming data (which includes arities) */
   for( i = 0; i < nvars; i++)
   {
      arities[i] = data[k++];
      data2[i] = mydata+j;
      for( l = 0; l < ndatapoints; l++)
         mydata[j++] = data[k++];
   }

   /* for( i = 0; i < nrows; i++) */
   /* { */
   /*    for( j = 0; j < nvars; j++) */
   /*       printf("%d ",data2[i][j]); */
   /*    printf("\n"); */
   /* } */

   /* for( i = 0; i < nrows; i++) */
   /* { */
   /*    for( j = 0; j < nvars; j++) */
   /*       printf("%d ",data[i*nvars + j]); */
   /*    printf("\n"); */
   /* } */

   adtree = ret_adtree(
      rmin, adtreedepthlim, adtreenodeslim, (VARIABLE) nvars,
      (COUNT) ndatapoints_plus_one - 1, /* number of datapoints */
      data2, arities);

   adtree_etc->adtree = adtree;
   adtree_etc->data = data2;
   adtree_etc->arity = arities;
   adtree_etc->nvars = (VARIABLE) nvars;
   
   /* for( i = 0; i < nvars; i++) */
   /*    printf("%d %d\n",i,adtree_etc->arity[i]); */
   
   return (void *) adtree_etc;
}

/** Construct a flat contingency table from a leaflist
 *  ( contingency table must be for at least one variable )
 */
static
void makeflatcontableaf(
   const ROW *leaflist,        /**< datapoints for this query  */
   COUNT count,                /**< number of datapoints (in leaflist) */
   const VARIABLE *variables,  /**< variables in the contingency table (sorted) */
   const int *strides,         /**< stride sizes for each variable */
   int nvariables,             /**< number of variables in the contingency table */
   FLATCONTAB flatcontab,      /**< Contingency table initialised to zero */
   VALUE **data                /**< data[i][j] is the value of variable i in row j */
)
{
   COUNT j;
   int i;
   int k;
   ROW row;

   int stride0;
   const VALUE* data0;
   int stride1;
   const VALUE* data1;
   int stride2;
   const VALUE* data2;
   int stride3;
   const VALUE* data3;
   int stride4;
   const VALUE* data4;

   switch( nvariables )
   {
   case 1: 
   {
       stride0 = strides[0]; 
       data0 = data[variables[0]]; 

      for( j = 0; j < count; ++j )
      {
         flatcontab[stride0*data0[leaflist[j]]]++;
         /* printf("d1 %d\n",flatcontab[stride0*data0[leaflist[j]]]); */
      }
      break;
   }
   case 2:
   {
        stride0 = strides[0]; 
        data0 = data[variables[0]];
        stride1 = strides[1]; 
        data1 = data[variables[1]];

        
        for( j = 0; j < count; ++j )
        {
           row = leaflist[j];
           flatcontab[stride0*data0[row] + stride1*data1[row]]++;
           /* printf("d2 %d\n",flatcontab[stride0*data0[row] + stride1*data1[row]]); */
        }

        break;
   }
   case 3:
   {
        stride0 = strides[0]; 
        data0 = data[variables[0]];
        stride1 = strides[1]; 
        data1 = data[variables[1]];
        stride2 = strides[2]; 
        data2 = data[variables[2]];

        
        for( j = 0; j < count; ++j )
        {
           row = leaflist[j];
           flatcontab[stride0*data0[row] + stride1*data1[row] + stride2*data2[row]]++;
        }

        break;
   }
   case 4:
   {
        stride0 = strides[0]; 
        data0 = data[variables[0]];
        stride1 = strides[1]; 
        data1 = data[variables[1]];
        stride2 = strides[2]; 
        data2 = data[variables[2]];
        stride3 = strides[3]; 
        data3 = data[variables[3]];

        
        for( j = 0; j < count; ++j )
        {
           row = leaflist[j];
           flatcontab[stride0*data0[row] + stride1*data1[row] + stride2*data2[row] + stride3*data3[row]]++;
        }

        break;
   }
   case 5:
   {
        stride0 = strides[0]; 
        data0 = data[variables[0]];
        stride1 = strides[1]; 
        data1 = data[variables[1]];
        stride2 = strides[2]; 
        data2 = data[variables[2]];
        stride3 = strides[3]; 
        data3 = data[variables[3]];
        stride4 = strides[4]; 
        data4 = data[variables[4]];

        
        for( j = 0; j < count; ++j )
        {
           row = leaflist[j];
           flatcontab[stride0*data0[row] + stride1*data1[row] + stride2*data2[row] + stride3*data3[row] + stride4*data4[row]]++;
        }

        break;
   }
   default:
   {
      for( j = 0; j < count; ++j )
      {
         row = leaflist[j];
         i = 0;
         for(k = 0; k < nvariables; k++)
            i += strides[k]*data[variables[k]][row];
         flatcontab[i]++;
      }

      break;
   }
   }
}


/** Construct a flat contingency table from an adtree
 */
static
void makeflatcontab(
   const ADTREE *adtree,       /**< (Pointer to) the ADTREE */
   VARIABLE offset,            /**< Offset for first variable (to identify correct vary nodes) */
   const VARIABLE *variables,  /**< Variables in the sought contingency table (sorted) */
   const int *strides,         /**< stride sizes for each variable */
   int nvariables,             /**< Number of variables in the contingency table */
   FLATCONTAB flatcontab,      /**< Contingency table initialised with zeroes */
   VALUE **data,               /**< data[i][j] is the value of variable i in row j */
   const ARITY *arity          /**< arity[i] is the arity of variable i, */
)
{

   VARYNODE vn;
   FLATCONTAB flatcontabmcv;
   VALUE val;
   FLATCONTAB flatcontabval;
   int i;

   /* printf("foo %d\n",nvariables); */
   
   assert(adtree != NULL);
   assert(flatcontab != NULL);
   assert(data != NULL);
   assert(arity != NULL);
   assert(nvariables == 0 || variables != NULL);
   assert(nvariables == 0 || strides != NULL);
   assert(adtree->children == NULL || adtree->leaflist == NULL );

   if( nvariables == 0 )
   {
      flatcontab[0] = adtree->count;
      return;
   }

   if( adtree->leaflist != NULL )
   {
      /* construct contingency table directly from data in leaf list */
      makeflatcontableaf(adtree->leaflist, adtree->count, variables, strides, nvariables, flatcontab, data); 
      return;
   }

   /* find varynode for firstvar */
   vn = adtree->children[variables[0] - offset];
   
   flatcontabmcv = flatcontab+strides[0]*vn.mcv;

   /* make contingency table where variables[0] is marginalised away (and store at flatcontab + stride*vn.mcv) */
   makeflatcontab(adtree, offset, variables+1, strides+1, nvariables-1, flatcontabmcv, data, arity);

   for( val = 0; val < arity[variables[0]]; ++val )
   {
      /* if vn.children[val] == NULL then either val=vn.mcv or the contingency table for val is all zeroes
         so do nothing */
      if( vn.children[val] != NULL)
      {
         flatcontabval = flatcontab + strides[0]*val;
         makeflatcontab(vn.children[val], variables[0]+1, variables+1, strides+1, nvariables-1, flatcontabval, data, arity);
         /* subtract contingency table for val from that for mcv */
         for(i = 0; i < strides[0]; ++i)
         {
            flatcontabmcv[i] -= flatcontabval[i];
         }
      }
   }
}

void makecontab(
   void* v_adtree_etc,       /**< (Pointer to) the ADTREE_ETC */
   unsigned int* variables,           /**< Variables in the sought contingency table (sorted) */
   int nvariables,            /**< Number of variables in the contingency table */
   unsigned int* flatcontab,
   int flatcontabsize
)
{

   int* strides;
   int i;
   int fsize;
   
   ADTREE_ETC* adtree_etc = (ADTREE_ETC*) v_adtree_etc;

   assert(adtree_etc != NULL);
   assert(adtree_etc->adtree != NULL);
   assert(adtree_etc->data != NULL);
   assert(adtree_etc->arity != NULL);
   
   /* compute size for a flat contingency table (and associated 'strides') */
   /* this size already given and has been computed in Python */
   strides = (int *) malloc(nvariables * sizeof(int));
   fsize = 1;
   for( i = nvariables - 1; i >= 0; i--)
   {
      strides[i] = fsize;
      fsize *= adtree_etc->arity[variables[i]];
   }

   for(i = 0; i < flatcontabsize; i++)
      flatcontab[i] = 0;

   /* for(i = 0; i < flatcontabsize; i++) */
   /*    printf("foo %d ",flatcontab[i]); */
   /* printf("\n"); */

   
   /* printf("making flatcontab\n");  */
   /* print_adtree(adtree_etc->adtree,0,adtree_etc->nvars,adtree_etc->arity); */
   /* printf("adtree printing done\n"); */
   
   makeflatcontab(adtree_etc->adtree, 0, (VARIABLE*) variables, strides, nvariables, flatcontab,
      adtree_etc->data, adtree_etc->arity);

   /* printf("size = %d",flatcontabsize); */
   /* for(i = 0; i < flatcontabsize; i++) */
   /*    printf("foo %d ",flatcontab[i]); */
   /* printf("\n"); */
   
   free(strides);

}


/** Delete an AD tree */
void delete_adtree(
   ADTREE *adtree,          /**< pointer to ADTREE being deleted */
   const VARIABLE variable, /**< first variable to specialise further on, if variable=nvars then there is none */
   VARIABLE nvars,          /**< Number of variables in the data */
   ARITY *arity             /**< arity[i] is the arity of variable i, */
)
{

   VARIABLE var;
   VARYNODE *vn_ptr = adtree->children;
   VARYNODE vn;
   VALUE val;

   if( adtree->leaflist != NULL )
   {
      assert(vn_ptr == NULL);
      free(adtree->leaflist);
   }
   else if( vn_ptr != NULL )
   {
      assert(adtree->leaflist == NULL);
      for( var = variable; var < nvars; ++var )
      {
         vn = vn_ptr[var - variable];
         for( val = 0; val < arity[var]; ++val )
            if( vn.children[val] != NULL )
               delete_adtree(vn.children[val], var + 1, nvars, arity);
         free(vn.children);
      }
      free(vn_ptr);
   }
   free(adtree);
   return;
}

/** Delete an AD tree */
void del_adtree(
   void *adtree,          /**< pointer to ADTREE being deleted */
   int nvars,          /**< Number of variables in the data */
   int *arity             /**< arity[i] is the arity of variable i, */
)
{
   delete_adtree((ADTREE*) adtree, 0, (VARIABLE) nvars, (ARITY*) arity);
}

