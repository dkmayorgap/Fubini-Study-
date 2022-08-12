"""
utils.py
--------
"""

import operator
from functools import reduce

import numpy as np
from math import factorial

def prod(a):
    """
    Return product of elements of a. Start with int 1 so if only ints are included then an int result is
    returned.
    """
    return reduce(operator.mul, a, 1)


def eijk(args):
    """
    Represent the Levi-Civita symbol. For even permutations of indices it returns 1, for odd
    permutations -1, and for everything else (a repeated index) it returns 0. Thus it represents an
    alternating pseudotensor.

    Parameters
    ----------
    args : tuple with int
        A tuple with indices.

    Returns
    -------
    int
    """
    n = len(args)

    return prod(prod(args[j] - args[i] for j in range(i + 1, n)) / factorial(i) for i in range(n))
