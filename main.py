import numpy as np
import tensorflow as tf
from models import ProjectiveSpace
from monte_carlo_integrate import Integrate

import matplotlib.pyplot as plt
import time


def testChernClasses(num_tests = 3, num_pts = 1000):
    """
    @param num_tests: Number of chern classes to test (currently only implemented upto c3).
    """
    for n in range(1, num_tests+1):
        x, p = Integrate.generateDomain(lambda x: True, -np.pi/2, np.pi/2, 2*n, num_pts)
        cpn = ProjectiveSpace(n)
        metric = cpn.getMetric(np.tan(x))
        riem = cpn.getRiemann(np.tan(x))

        norm_coeff = np.prod([np.power(1./np.cos(x[:, i]), 2) for i in range(x.shape[1])], axis=0)
        weights = p*norm_coeff

        print(f"Integral of the metric = {np.mean(np.real(np.linalg.det(metric))*weights)} on CP{n}. Should be {np.power(np.pi,n) / np.math.factorial(n)}")

        if n == 1:
            c1 = ProjectiveSpace.getC1(riem)
            c1_top = ProjectiveSpace.getTopNumber(c1)
            print(f"int_CP1 c1 = {np.real(np.mean(c1_top*weights)*((-2*1j)**n))}. Should be {n}+1 = 2")
        elif n == 2:
            c1 = ProjectiveSpace.getC1(riem)
            c2 = ProjectiveSpace.getC2(riem)

            c1c1_top = ProjectiveSpace.getTopNumber(
                ProjectiveSpace.computeProduct(c1, c1))
            c2_top = ProjectiveSpace.getTopNumber(c2)

            print(f"int_CP2 c1 ^ c1 = {np.real(np.mean(weights * c1c1_top) * (-2*1j)**n)}. Should be ({n}+1)^{n} = 9")
            print(f"int_CP2 c2 = {np.real(np.mean(weights * c2_top) * (-2*1j)**n)}. Should be {n}+1 = 3")
        elif n == 3:
            c1 = ProjectiveSpace.getC1(riem)
            c2 = ProjectiveSpace.getC2(riem)
            c3 = ProjectiveSpace.getC3(riem)

            c1c1c1_top = ProjectiveSpace.getTopNumber(
                ProjectiveSpace.computeProduct(
                    ProjectiveSpace.computeProduct(c1, c1), c1))
            c1c2_top = ProjectiveSpace.getTopNumber(
                ProjectiveSpace.computeProduct(c1, c2))
            c3_top = ProjectiveSpace.getTopNumber(c3)

            print(f"int_CP3 c1 ^ c1 ^ c1 = {np.real(np.mean(weights * c1c1c1_top) * (-2*1j)**n)}. Should be ({n}+1)^{n} = 64.")
            print(f"int_CP3 c1 ^ c2 = {np.real(np.mean(weights * c1c2_top) * (-2*1j)**n)}. Should be 24.")
            print(f"int_CP3 c3 = {np.real(np.mean(weights * c3_top)* (-2*1j)**n)}. Should be 4.")



def scalar():
    for n in range(2, 5):
        proj = ProjectiveSpace(n)
        x, _ = Integrate.generateDomain(lambda x: True, -int(10), int(10), 2*n, 1)
        points = tf.Variable(x, tf.float32)
        print(points)

        riemm = proj.getRiemann(points)
        metric = proj.getMetric(points)
        g_inv = np.linalg.inv(metric)

        ricci = tf.einsum("xabya->xby", riemm)
        r_scalar = tf.einsum("xab,xab", ricci, g_inv)
        print(f"{r_scalar} = {n}({n}+1) = {n*(n+1)}")


if __name__ == "__main__":
    #scalar()
    testChernClasses()
