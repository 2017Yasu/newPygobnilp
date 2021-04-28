from gurobipy import GRB

def obligatory_arrows(gobnilp):
    return [('A','E')]

def mipconss(gobnilp):
    orvar = gobnilp.addVar(obj=-1000,vtype=GRB.BINARY)
    gobnilp.addGenConstrOr(orvar, [gobnilp.adjacency[frozenset(['A','B'])],
                                    gobnilp.adjacency[frozenset(['A','F'])],
                                    gobnilp.adjacency[frozenset(['D','E'])]])
