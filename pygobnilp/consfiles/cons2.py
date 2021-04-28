def obligatory_arrows(gobnilp):
    return ([(v,'E') for v in gobnilp.bn_variables if v != 'E'])
