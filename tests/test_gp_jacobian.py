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

jac_mean, jac_cov = manifold.embed_jac(Xnew)
print(jac_mean.shape)
print(jac_cov.shape)
print(jac_mean)
print(jac_cov)
