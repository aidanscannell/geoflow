#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from geoflow.gp import gp_predict_jacobian
from gpflow.config import default_float

# from absl.testing import parameterized
from gpflow.kernels import SeparateIndependent, SquaredExponential
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Constant
from gpflow.models import SVGP

rng = np.random.RandomState(1)

# test inputs
num_data = 10
input_dim = 2
output_dim = 1
num_inducing = 3
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
# inducing_variable = rng.randn(num_inducing, input_dim)
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

# gp_predict_jacobian(
#     Xnew,
#     svgp.inducing_variable.Z,
#     svgp.kernel,
#     svgp.mean_function,
#     svgp.q_mu,
#     full_cov=False,
#     q_sqrt=svgp.q_sqrt,
#     whiten=svgp.whiten,
# )

svgp.predict_jacobian(Xnew)
