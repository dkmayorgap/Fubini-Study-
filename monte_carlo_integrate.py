"""
monte_carlo_integrate.py
------------------------
"""

import numpy as np


# TODO: delegate computation of integration weights to this class
class Integrate(object):
    @staticmethod
    def generateDomain(domain, a, b, n, num_pts):
        x_unif = np.random.uniform(a, b, (num_pts, n))
        filter_domain = [domain(x) for x in x_unif]

        p = np.power(b - a, n, dtype=np.float64) * sum(filter_domain) / len(filter_domain)

        return x_unif[filter_domain], p

    @staticmethod
    def integrate(func, domain, a, b, n, num_pts):
        """
        integrate

        :param func: real/complex valued function.
        :param domain: characteristic function.
        :param a: domain [a,b]^n specification variable.
        :param b: domain [a,b]^n specification variable.
        :param n: domain [a,b]^n specification variable.
        :param num_pts: number of points to sample.
        """
        x, p = Integrate.generateDomain(domain, a, b, n, num_pts)

        y = func(x)
        y_exp = y.sum() / len(y)
        return p * y_exp

