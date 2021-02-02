from pruners.lasso import lasso_pruning
from pruners.norm import l1norm_pruning

pruner_methods = \
{
    "lasso": lasso_pruning,
    "l1norm": l1norm_pruning
}


def get_pruner(name_str):
    if name_str in pruner_methods.keys():
        return pruner_methods[name_str]
    else:
        print("pruner {} is not supported".format(name_str))
        return None