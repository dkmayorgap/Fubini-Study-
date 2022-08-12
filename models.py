"""
models.py
---------
"""

import tensorflow as tf
import numpy as np

import itertools
from utils import eijk


class ProjectiveSpace(object):
    def __init__(self, ndims, homogeneous=False):
        """
        @param homogeneous: Specificies if the coordinates are homogeneous.
        @param ndims: complex dimension.
        """
        self.ndims = ndims + 1 if homogeneous else ndims
        self.homogeneous = homogeneous

    @tf.function
    def getKahlerPotential(self, points):
        assert points.shape[1] == 2*self.ndims
        x = points[:, :self.ndims]
        y = points[:, self.ndims:]

        c = 0 if self.homogeneous else 1
        return tf.math.log(tf.math.reduce_sum(tf.math.pow(x, 2), axis=1) + tf.math.reduce_sum(tf.math.pow(y, 2), axis=1) + c)

    @tf.function
    def getMetric(self, points):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(points)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(points)
                phi = self.getKahlerPotential(points)
            di_phi = tape2.gradient(phi, points)
        dij_phi = tape1.batch_jacobian(di_phi, points)

        g_real = 0.25 * dij_phi[:, :self.ndims, :self.ndims] # xx
        g_real += 0.25 * dij_phi[:, self.ndims:, self.ndims:] # yy
        g_imag = 0.25 * dij_phi[:, :self.ndims, self.ndims:] # xy
        g_imag += -0.25 * dij_phi[:, self.ndims:, :self.ndims] # yx

        return tf.complex(g_real, g_imag)


    @tf.function
    def getChristoffel(self, points):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(points)
            g = self.getMetric(points)
            g_real = tf.math.real(g)
            g_imag = tf.math.imag(g)

        di_g = tf.complex(
            tape.batch_jacobian(g_real, points),
            tape.batch_jacobian(g_imag, points))

        gij_k = di_g[:, :, :, :self.ndims]*0.5 # dxk
        gij_k -= 1j*di_g[:, :, :, self.ndims:]*0.5 # dyk

        g_inverted = tf.linalg.inv(tf.complex(g_real, g_imag))

        return tf.einsum("xlk,xjli->xijk", g_inverted, gij_k)


    @tf.function
    def getRiemann(self, points):
        return ProjectiveSpace.getRiemannPB(
            points, lambda x: tf.cast(self.getMetric(x), tf.complex128),
            lambda x: tf.cast(np.asarray([np.identity(self.ndims) for _ in range(x.shape[0])]), tf.complex128), self.ndims)

    @staticmethod
    @tf.function
    def getRiemannPB(input_tensor, metric_func, pullbacks_func, ndims):
        """
        @param input_tensor: tensorflow input variable for points. Shape: (N, 2*ndims).
        @param metric_func: function to compute metric. tf.function. Shape: (N, ncydims, ncydims).
        @param pullbacks_func: pullbacks for transforming from CPn to CY. tf.function. Shape: (N, ncydims, ndims)
        """
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(input_tensor)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(input_tensor)

                g_pred = metric_func(input_tensor)
                g_pred_real = tf.math.real(g_pred)
                g_pred_imag = tf.math.imag(g_pred)
            dg_dZ_real = tape2.batch_jacobian(g_pred_real, input_tensor)
            dg_dZ_imag = tape2.batch_jacobian(g_pred_imag, input_tensor)

            pullbacks = pullbacks_func(input_tensor)
            dg_dZ = tf.complex(dg_dZ_real, dg_dZ_imag)
            dg_dZ_pb = tf.einsum("xbdi,xai->xabd", 0.5*(dg_dZ[:, :, :, :ndims] -(1j*dg_dZ[:, :, :, ndims:])), pullbacks)
            g_pred = metric_func(input_tensor)
            g_pred_inv = tf.math.conj(tf.linalg.inv(g_pred))
            Gamma = tf.einsum("xabd,xyd->xaby", dg_dZ_pb, g_pred_inv)
            Gamma_real = tf.math.real(Gamma)
            Gamma_imag = tf.math.imag(Gamma)

        dGamma_real = tape1.batch_jacobian(Gamma_real, input_tensor)
        dGamma_imag = tape1.batch_jacobian(Gamma_imag, input_tensor)

        dGamma = tf.complex(dGamma_real, dGamma_imag)
        dGamma_pb = (dGamma[:, :, :, :, :ndims] + (1j*dGamma[:, :, :, :, ndims:]))*0.5
        dGamma_pb = tf.einsum("xaydi,xbi->xabyd", dGamma_pb, tf.math.conj(pullbacks))
        Riem = -dGamma_pb

        return Riem

    @staticmethod
    def _generateSumString(n):
        chars = "abcdefghijklmnopqrstuvwyz"
        sumString = []
        finalIndices = []
        for i in range(n):
            ki = 3*i + 0
            li = 3*i + 1
            ji = 3*i + 2
            kip1 = 3*i + 3
            sumString.append("x"+chars[ki] + chars[li] + chars[ji] + chars[kip1])
            finalIndices.append(chars[li] + chars[ji])
        sumString[-1] = sumString[-1][:-1] + sumString[0][1]
        contraction = ','.join(sumString)
        return f"{contraction}->x{''.join(finalIndices)}"

    @staticmethod
    def _generateSeparateTraceSumString(n):
        chars = "abcdefghijklmnopqrstuvwyz"
        sumString = []
        finalIndices = []
        for i in range(n):
            ki = 3*i + 0
            li = 3*i + 1
            ji = 3*i + 2
            sumString.append("x" + chars[ki] + chars[li] + chars[ji] + chars[ki])
            finalIndices.append(chars[li] + chars[ji])
        contraction = ','.join(sumString)
        return f"{contraction}->x{''.join(finalIndices)}"

    @staticmethod
    def traceOfRiemann(riem, n):
        """
        Computes Tr(R^n).
        :param n: power of the riem.
        """
        sumString = ProjectiveSpace._generateSumString(n)
        print(f"sumString={sumString}")
        return tf.einsum(sumString, *([riem]*n))

    @staticmethod
    def computeProduct(a, b):
        """
        Computes product:
            (x, a1, ..., an), (x, b1, ..., bn) -> (x, a1, ..., an, b1, ..., bn)
        """
        return tf.convert_to_tensor([tf.tensordot(a[i,:], b[i,:], axes=0) for i in range(a.shape[0])], dtype=tf.complex128)

    @staticmethod
    def traceOfRiemannPower(riem, n):
        """
        Computes (TrR)^n
        :param n: power of trace.
        """
        sumString = ProjectiveSpace._generateSeparateTraceSumString(n)
        print(f"sumStringSeparate={sumString}")
        return tf.einsum(sumString, *([riem]*n))


    @staticmethod
    def getTopNumber(tensor):
        """
        Given a top complex form (n,n):
            omega = A_r1s1r2s2...rnsn dzr1 ^ dzbs1 ^ ... ^ dzrn ^ dzbsn
        computes coefficient c s.t.:
            omega = c vol = c dz1 ^ dzb1 ^ ... ^ dzn ^ dzbn

        :param tensor: Shape len(num_pts,n,  n, ... , n, n) = 2*n + 1
                                    x ,  r1, s1, ..., rn, sn) = 2*n + 1
        """
        # TODO: more assertion tests
        assert (len(tensor.shape) - 1) % 2 == 0
        n = tensor.shape[1]
        assert ((len(tensor.shape) - 1)//2) == n

        c = 0
        for r_prod in itertools.product(*[range(n) for _ in range(n)]):
            for s_prod in itertools.product(*[range(n) for _ in range(n)]):
                # r_prod and s_prod have to be interlaced, with (r_prod[0], s_prod[0], r_prod[1], s_prod[1], ..., r_prod[n-1], s_prod[n-1])
                indices = [r_prod[int(i / 2)] if i % 2 == 0 else s_prod[int((i - 1)/2)] for i in range(2*n)]
                c += tensor[tuple([...] + indices)]*eijk(r_prod)*eijk(s_prod)
        return c

    @staticmethod
    def getC1(riem):
        return (1j / (2*np.pi)) * ProjectiveSpace.traceOfRiemann(riem, 1)

    @staticmethod
    def getC2(riem):
        TrRwTrR = ProjectiveSpace.traceOfRiemannPower(riem, 2)
        TrR2 = ProjectiveSpace.traceOfRiemann(riem, 2)
        return 0.5 * (1 / np.power(2*np.pi, 2)) * (TrR2 - TrRwTrR)

    @staticmethod
    def getC3(riem):
        TrR2 = ProjectiveSpace.traceOfRiemann(riem, 2)
        TrR3 = ProjectiveSpace.traceOfRiemann(riem, 3)
        c1 = ProjectiveSpace.getC1(riem)
        c2 = ProjectiveSpace.getC2(riem)

        term1 = ProjectiveSpace.computeProduct(c1, c2)/3.0
        term2 = (1.0/3.0)*(1 / np.power(2*np.pi, 2)) * ProjectiveSpace.computeProduct(c1, TrR2)
        term3 = - (1.0/3.0) * (1j / np.power(2*np.pi, 3)) * TrR3
        print(f"Shapes are: t1={term1.shape}, t2={term2.shape}, t3={term3.shape}")

        return term1 + term2 + term3

