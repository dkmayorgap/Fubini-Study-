# FubiniStudy

This repository contains code for computing Riemann tensor and chern classes for any hermitian metric $g$ on some manifold $M$ of dimension $2n$.

Locally the components of $g$ are given by:
$$\tag{1.1}g = g_{i\overline{j}} dz^i \wedge d\overline{z}^j$$


## models.ProjectiveSpace
The repository contains sample implementation of the Fubini-Study metric $g_\text{FS}$ for any $\mathbb{CP}^n$, which is defined as:
$$\tag{1.2}g_\text{FS} = \partial\overline{\partial}\ln|Z|^2\quad\text{or}\quad g_\text{FS} = \partial\overline{\partial}\ln(1+|z|^2)$$
where $[Z_0:\dots:Z_n]$ are the homogeneous coordinates on $\mathbb{CP}^n$ and $\{z_1,\cdots, z_n\}$ are defined on the patch $U_0 = \{Z\in\mathbb{CP}^n~\vert~Z_0 \neq  0\}$.

As an example, to initialize a complex projective space of dimension $n$ and to compute FubiniStudy metric and the corresponding Riemann tensor, use:
```python
from models import ProjectiveSpace
cpn = ProjectiveSpace(n, homogeneous = True)

# The shape of point is: [x0, x1, ..., xn, y0, ..., yn]
# where Zk = xk + i*yk.
# here points is an array of points. Shape: (N, 2*n) where N is the number of points.
metric = cpn.getMetric(points)
riemann = cpn.getRiemann(points)
```

## Chern Classes
The library provides an ability to compute various combinations of the Chern classes derived from the Riemann tensor. Currently only $c_1$, $c_2$ and $c_3$ are implemented. The expression for each is given by the following:
$$c_1 = \frac{i}{2\pi}\mathrm{Tr}(R)$$
$$c_2 = \frac{1}{(2\pi)^2}\frac{1}{2}\left[\mathrm{Tr}(R^2) - \mathrm{Tr}(R)^2\right]$$
$$c_3 = \frac{1}{3}\left[c_1\wedge c_2 + \frac{1}{(2\pi)^3}c_1\wedge \mathrm{Tr}(R^2)- \frac{i}{(2\pi)^3}\mathrm{Tr}(R^3)\right]$$
For more detailed reference, see [1]. Each can be computed in the following manner:
```python
# Here riem is the Riemann tensor
c1 = ProjectiveSpace.getC1(riem)
c2 = ProjectiveSpace.getC2(riem)
c3 = ProjectiveSpace.getC3(riem)
```
To compute products $c_i \wedge c_j$ use:
```python
cicj = ProjectiveSpace.computeProduct(ci, cj)
```
For expressions of form $\mathrm{Tr}(R^n)$ use:
```python
TrRn = ProjectiveSpace.traceOfRiemann(riem, n)
```
and to compute expressions $\mathrm{Tr}(R)^n$ use:
```python
TrR_n = ProjectiveSpace.traceOfRiemannPower(riem, n)
```


## Integrating Forms
### Converting top-forms to number arrays
Suppose we take a manifold $M$ of complex dimension $n$. Suppose we have a top form $A\in \Omega^{(n,n)}(M)$, represented as a `np.array`/`tf.tensor` `A` of shape `(N, n, n, ...,, n, n)` where we have total of `n` pairs of `(n,n)`, that is: the rank of `A` is $2n+1$. Since $A$ is a top form, we may locally (on $U\subset M$) represent $A$ as:
$$\tag{2.1}A = c dz^1\wedge d\overline{z}^1 \wedge \cdots \wedge dz^{n}\wedge d\overline{z}^n$$
for some function $c\colon U\rightarrow \mathbb{C}$. To compute $c$ from `A`, use the following:
```python
c = ProjectiveSpace.getTopNumber(A)
```
for e.g. suppose we have $M = \mathbb{CP}^3$, then, $c_3$ is a top form on $M$, thus we can compute this using:
```python
cp3 = ProjectiveSpace(3)
...
c3 = ProjectiveSpace.getC3(riem)
c3_top = ProjectiveSpace.getTopNumber(c3)
```
### Monte-Carlo integration
Since the input variables are real, before integrations, we have to perform pullbacks using canonical identification $\phi\colon \mathbb{R}^2\rightarrow \mathbb{C}$.
As an example suppose we have $A$ defined as in (2.1), then:
$$\phi^* A = (-2i)^ncdx^1\wedge dy^1\wedge \cdots\wedge dx^n \wedge dy^n$$

Thus, as an example, to integrate $A$ over $\mathbb{CP}^n$, one can use the `monte_carlo_integrate` module:
```python
from monte_carlo_integrate import Integrate

x, p = Integrate.generateDomain(lambda x: True, -np.pi/2, np.pi/2, 2*n, num_pts)
cpn = ProjectiveSpace(n)
# Compute form A on points x
...

c = ProjectiveSpace.getTopNumber(A)

norm_coeff = np.prod([np.power(1./np.cos(x[:, i]), 2) for i in range(x.shape[1])], axis=0)
weights = p*norm_coeff
norm_coeff = (-2*1j)**n
print(f"int_CPn A = {np.real(np.mean(weights * c * norm_coeff))}")
```
> Note: this chooses a chart where $Z_0 \neq 0$. By default `ProjectiveSpace(n)` uses inhomogeneous coordinates in this chart.

For computation of the chern numbers see [test_integrate.ipynb](https://github.com/gbutb/FubiniStudy/blob/main/test_integrate.ipynb).

## Other metrics/manifolds
Using `ProjectiveSpace.getRiemannPB` it is possible to compute Riemann tensor for an arbitrary Calabi-Yau manifold $M$ using metric from [2]. See [CalabiYau/](https://github.com/gbutb/FubiniStudy/tree/main/CalabiYau) for details.

## References:
[1] - Bonetti, F., Weissenbacher, M. The Euler chern correction to the Kähler potential — revisited. J. High Energ. Phys. 2017, 3 (2017). https://doi.org/10.1007/JHEP01(2017)003

[2] - Learning Size and Shape of Calabi-Yau Spaces  Magdalena Larfors (Uppsala U.), Andre Lukas (Oxford U., Theor. Phys.), Fabian Ruehle (Northeastern U.), Robin Schneider (Uppsala U.)  e-Print: 2111.01436 [hep-th]
