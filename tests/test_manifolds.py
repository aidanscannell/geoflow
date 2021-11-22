#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# from absl.testing import parameterized
from gpflow.kernels import SeparateIndependent, SquaredExponential
from gpflow.likelihoods import Gaussian
from gpflow.inducing_variables import InducingPoints
from gpflow.mean_functions import Constant
from gpflow import default_float
from geoflow.gp import SVGP

# from gpflow.models import SVGP
from geoflow.manifolds import GPManifold

rng = np.random.RandomState(1)

# test inputs
num_data = 10
input_dim = 2
output_dim = 1
num_inducing = 30
Xnew = tf.constant(rng.randn(num_data, input_dim), dtype=default_float())
print(Xnew.shape)


# @dataclass(frozen=True)
# class DatumSVGP:
#     rng: np.random.RandomState = np.random.RandomState(0)
#     X = rng.randn(num_data, input_dim)
#     Y = rng.randn(num_data, output_dim) ** 2
#     Z = rng.randn(num_inducing, input_dim)
#     qsqrt = (rng.randn(num_inducing, num_output) ** 2) * 0.01
#     qmean = rng.randn(num_inducing, num_output)
#     lik = gpflow.likelihoods.Exponential()
#     data = (X, Y)


# default_datum_svgp = DatumSVGP()

# SVGP variants
if output_dim > 1:
    kernels = [
        SquaredExponential(lengthscales=np.ones(input_dim), variance=2.0)
        for _ in range(output_dim)
    ]
    kernel = SeparateIndependent(kernels)
else:
    # kernel = SquaredExponential(lengthscales=np.ones(input_dim), variance=2.0)
    kernel = SquaredExponential()
inducing_variable = InducingPoints(rng.randn(num_inducing, input_dim))

svgp = SVGP(
    kernel,
    likelihood=Gaussian(),
    inducing_variable=inducing_variable,
    mean_function=Constant(),
    num_latent_gps=output_dim,
    q_diag=False,
    whiten=False,
)
print(svgp.kernel.lengthscales)

manifold = GPManifold(gp=svgp)


def test_energy():
    """Check shapes of output"""

    f_mean, f_cov = manifold.embed(Xnew)


def test_embed():
    """Check shapes of output"""

    f_mean, f_cov = manifold.embed(Xnew)
    print(f_mean.shape)
    print(f_cov.shape)
    jac_mean, jac_cov = manifold.embed_jac(Xnew)
    print(jac_mean.shape)
    print(jac_cov.shape)
    print(jac_mean)
    print(jac_cov)

    # if num_data is not None:
    #     assert inner_prod.shape[0] == num_data
    # else:
    #     assert inner_prod.shape == ()

    # def geodesic_ode(pos, vel):
    #     return manifold.geodesic_ode(pos, vel)

    # var_geodesic_ode = self.variant(geodesic_ode)
    # state_prime = var_geodesic_ode(Xnew, u)
    # if num_data is None:
    #     assert state_prime.ndim == 1
    #     assert state_prime.shape[0] == 2 * input_dim
    # else:
    #     assert state_prime.ndim == 2
    #     assert state_prime.shape[0] == num_data
    #     assert state_prime.shape[1] == 2 * input_dim


test_embed()
# test_energy()

# @parameterized.product(num_data=num_datas)
# def test_inner_product(self, num_data):
#     """Check shapes of output"""
#     if num_data is not None:
#         u = np.ones([num_data, input_dim])
#         v = np.ones([num_data, input_dim])
#         Xnew = np.random.uniform((num_data, input_dim))
#     else:
#         u = np.ones([input_dim])
#         v = np.ones([input_dim])
#         Xnew = np.random.uniform(input_dim,))

#     def inner_product(Xnew, u, v):
#         return manifold.inner_product(Xnew, u, v)

#     var_inner_product = self.variant(inner_product)
#     inner_prod = var_inner_product(Xnew, u, v)
#     if num_data is not None:
#         assert inner_prod.shape[0] == num_data
#     else:
#         assert inner_prod.shape == ()

#     def geodesic_ode(pos, vel):
#         return manifold.geodesic_ode(pos, vel)

#     var_geodesic_ode = self.variant(geodesic_ode)
#     state_prime = var_geodesic_ode(Xnew, u)
#     if num_data is None:
#         assert state_prime.ndim == 1
#         assert state_prime.shape[0] == 2 * input_dim
#     else:
#         assert state_prime.ndim == 2
#         assert state_prime.shape[0] == num_data
#         assert state_prime.shape[1] == 2 * input_dim
