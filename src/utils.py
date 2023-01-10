import random
from enum import Enum, auto



class OptmAlgo(Enum):
    """Optimization Algorithms Available."""
    
    # Batch Gradient Descent
    BGD = auto()
    # Stochastic Gradient Descent
    SGD = auto()
    # Mini-Batch Gradient Descent
    MBGD = auto()
    # Gradient Descent with momentum
    # Adam Optimization
    # BFGS Optimization


def shuffled_seq_index(m: int, iter: int) -> list:
    """Generates a list of :iter random indexes (from 0 to :m).
    the generated list have a sequence of indexes (from 0 to :m)
    that gets repeated every :m element.

    :param m: number of indexes that gets repeated
    :param iter: length of the final list
    :return: list of shuffeled int indexes
    """
    if not (isinstance(m,int) and isinstance(iter,int)):
        raise TypeError("The function 'shuffled_seq_index' requires only integer arguments")

    gen_id = []
    for _ in range(iter//m):
        ind = list(range(m))
        random.shuffle(ind)
        gen_id.extend(ind)
    if iter%m != 0:
        ind = list(range(iter%m))
        random.shuffle(ind)
        gen_id.extend(ind)

    return gen_id


def argsort(arr1, arr2):
    """Sort :arr2 using sorted indexes from :arr1.

    :param arr1: array to sort and extract sorted indexes from
    :param arr2: array to sort using the sorted :arr1 as a reference
    :return: tuple of 2 sorted arrays
    """
    # get a list of sorted indexes for :arr1
    indexes = arr1.argsort()

    # sort :arr1 and :arr2 using the sorted indexes from :arr1 as reference
    return arr1[indexes], arr2[indexes]
