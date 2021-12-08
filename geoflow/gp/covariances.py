#!/usr/bin/env python3
import tensorflow as tf
from gpflow.config import default_float
from gpflow.inducing_variables import InducingPoints, SharedIndependentInducingVariables
from gpflow.kernels import SeparateIndependent, SquaredExponential
from gpflow.utilities import Dispatcher

hess_Kuu_wrt_Xnew = Dispatcher("hess_Kuu_wrt_Xnew")
jac_Kuf_wrt_Xnew = Dispatcher("jac_Kuf_wrt_Xnew")


@jac_Kuf_wrt_Xnew.register(object, object, InducingPoints)
@jac_Kuf_wrt_Xnew.register(SquaredExponential, object, InducingPoints)
def _jac_Kuf_wrt_Xnew(kernel, Xnew, inducing_variable):
    return jac_Kuf_wrt_Xnew(kernel, Xnew, inducing_variable.Z)


@jac_Kuf_wrt_Xnew.register(SquaredExponential, object, object)
def _jac_Kuf_wrt_Xnew(kernel, Xnew, X):
    num_test, _ = Xnew.shape
    num_data, input_dim = X.shape

    ksx = kernel.K(Xnew, X)

    X = tf.concat(num_test * [tf.expand_dims(X, 0)], 0)
    Xnew = tf.transpose(tf.concat(num_data * [tf.expand_dims(Xnew, 0)], 0), [1, 0, 2])

    jac_1 = Xnew - X
    jac_2 = tf.transpose(tf.concat(input_dim * [tf.expand_dims(ksx, 0)], 0), [1, 2, 0])
    jac = -(kernel.lengthscales ** -2) * jac_1 * jac_2
    print(f"jac.shape: {jac.shape}")
    return tf.expand_dims(jac, 1)


# Autodiff Jacobian works
@jac_Kuf_wrt_Xnew.register(object, object, object)
def jac_Kuf_wrt_Xnew_autodiff_via_distance(kernel, Xnew, X):
    """Kernel Jacobian dk_dx1 via partial derivatives of distance using autodiff"""
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t_x1:
        t_x1.watch(Xnew)
        dist = kernel.scaled_squared_euclid_dist(Xnew, X)
    ddist_dx = t_x1.batch_jacobian(dist, Xnew)

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t_dist:
        t_dist.watch(dist)
        k = kernel.K_r2(dist)
    dk_ddist = t_dist.batch_jacobian(k, dist)

    jac = tf.einsum("bik,bkm->bim", dk_ddist, ddist_dx)
    print(f"jac.shape: {jac.shape}")
    return tf.expand_dims(jac, 1)


@jac_Kuf_wrt_Xnew.register(
    SeparateIndependent, object, SharedIndependentInducingVariables
)
def _jac_Kuf_wrt_Xnew(kernel, Xnew, inducing_variable):
    return tf.concat(
        [
            jac_Kuf_wrt_Xnew(k, Xnew, inducing_variable.inducing_variable)
            for k in kernel.kernels
        ],
        axis=1,
    )


@hess_Kuu_wrt_Xnew.register(SquaredExponential, object, InducingPoints)
def _hess_Kuu_wrt_Xnew(kernel, Xnew, inducing_variable, full_cov=True):
    return hess_Kuu_wrt_Xnew(kernel, Xnew, inducing_variable.Z, full_cov=full_cov)


@hess_Kuu_wrt_Xnew.register(SquaredExponential, object, object)
def _hess_Kuu_wrt_Xnew(kernel, Xnew, X, full_cov=True):
    num_test, input_dim = Xnew.shape

    const = kernel.lengthscales ** -2 * kernel.variance
    if full_cov:
        hess = const * tf.eye(input_dim, dtype=default_float())
    else:
        hess = const * tf.ones(input_dim, dtype=default_float())
    hess = tf.broadcast_to(hess, [num_test, 1, *hess.shape])
    return hess


@hess_Kuu_wrt_Xnew.register(
    SeparateIndependent, object, SharedIndependentInducingVariables
)
def _hess_Kuu_wrt_Xnew(kernel, Xnew, inducing_variable, full_cov=True):
    return tf.concat(
        [
            hess_Kuu_wrt_Xnew(
                k, Xnew, inducing_variable.inducing_variable, full_cov=full_cov
            )
            for k in kernel.kernels
        ],
        axis=1,
    )


# Old hessian covariance (hess doesn't work with autodiff!!!)


def grad_kernel_wrt_x1x1_via_distance(kernel, x1):
    """Kernel Hessian d2k_dx1dx1 via partial derivatives of distance"""
    # TODO: autodiff doesn't seem to be able to do this...
    x2 = x1
    # Derivatives of squared euclidean distance wrt inputs
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t_x1:
        t_x1.watch(x1)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t_x2:
            t_x2.watch(x2)
            dist = kernel.scaled_squared_euclid_dist(x1, x2)
        ddist_dx = t_x2.batch_jacobian(dist, x2)
    d2dist_d2x = t_x1.batch_jacobian(ddist_dx, x1)

    # Derivatives of kernel wrt squared euclidean distance
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t_dist_2:
        t_dist_2.watch(dist)
        with tf.GradientTape(
            persistent=True, watch_accessed_variables=False
        ) as t_dist_1:
            t_dist_1.watch(dist)
            k = kernel.K_r2(dist)
        dk_ddist = t_dist_1.batch_jacobian(k, dist)
    d2k_d2dist = t_dist_2.batch_jacobian(dk_ddist, dist)

    # Combine partial derivatives
    hess_2 = tf.einsum("bik,bkmn->bimn", dk_ddist, d2dist_d2x)
    # print(f"hess_2.shape: {hess_2.shape}")
    hess_1 = tf.einsum("bikl,bkm,bln->bimn", d2k_d2dist, ddist_dx, ddist_dx)
    # print(f"hess_1.shape: {hess_1.shape}")
    hess = hess_1 + hess_2
    # print(f"hess.shape: {hess.shape}")
    return hess
