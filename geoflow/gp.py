#!/usr/bin/env python3
from functools import partial
from typing import Optional, Tuple

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow.conditionals import base_conditional, conditional
from gpflow.conditionals.util import separate_independent_conditional_implementation
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import SVGP
from gpflow.types import MeanAndVariance

from geoflow.conditionals import covariance_conditional, svgp_covariance_conditional
from geoflow.custom_types import InputDim, NumData, One, OutputDim

InputData = None

JacMeanAndVariance = Tuple[
    ttf.Tensor3[NumData, InputDim, OutputDim],
    ttf.Tensor4[NumData, OutputDim, InputDim, InputDim],
]


def predict_jacobian(
    gp, Xnew: ttf.Tensor2[NumData, InputDim], full_cov: Optional[bool] = False
):
    jac_mean, jac_cov = gp_predict_jacobian(
        Xnew,
        gp.inducing_variable.Z,
        gp.kernel,
        gp.mean_function,
        f=gp.q_mu,
        full_cov=full_cov,
        q_sqrt=gp.q_sqrt,
        whiten=gp.whiten,
    )
    return jac_mean, jac_cov


SVGP.predict_jacobian = predict_jacobian


# # @tf.function()
# def gp_predict_jacobian(
#     Xnew: ttf.Tensor2[NumData, InputDim],
#     X: ttf.Tensor2[NumData, InputDim],
#     # inducing_variable: InducingVariables,
#     # X: InputData,
#     kernel: Kernel,
#     mean_function: MeanFunction,
#     f,
#     full_cov: Optional[bool] = False,
#     q_sqrt=None,
#     whiten: Optional[bool] = False,
# ) -> JacMeanAndVariance:
#     # X = inducing_variable.Z
#     jitter = default_jitter()
#     # print(jitter)
#     Kxx = kernel.K(X)
#     # Kxx = kernel.K(X, full_cov=False)
#     # Kxx = kernel(X, full_cov=False)
#     # Kxx += jitter * tf.eye(Kxx.shape[-1])
#     print("Kxx")
#     print(Kxx.shape)
#     # Kxx = tf.expand_dims(Kxx, 0)
#     # num_data = Xnew.shape[0]
#     # Kxx = tf.tile(Kxx, [num_data, 1, 1])
#     # print(Kxx.shape)

#     def jac_kern_fn_wrapper(x1):
#         # x1 = x1.reshape(1, -1)
#         K = kernel.K(x1, X)
#         return K

#     def hess_kern_fn_wrapper(x1, x2):
#         # K = kernel(x1, full_cov=False)
#         K = kernel.K(x1, x2)
#         # K = tf.linalg.diag_part(K)
#         # print("K hessdig")
#         # print(K.shape)
#         # print(K)
#         # return tf.reshape(K, [-1, 1])
#         return K

#     def hess_kern_fn_wrapper_2(x1):
#         return kernel.K(x1)

#     with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t2:
#         t2.watch(Xnew)
#         with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t1:
#             t1.watch(Xnew)
#             dk = jac_kern_fn_wrapper(Xnew)
#             # print("dk")
#             # print(dk)
#             # d2k = hess_kern_fn_wrapper(Xnew, Xnew)
#             d2k = hess_kern_fn_wrapper_2(Xnew)
#             # print("d2k")
#             # print(d2k)
#             # d2k = kernel.K(Xnew, Xnew)

#         jac = t1.batch_jacobian(dk, Xnew)
#         jh = t1.batch_jacobian(d2k, Xnew)
#     print(f"jac.shape: {jac.shape}")
#     # hess = t2.batch_jacobian(jh, Xnew)[:, 0, :, :]
#     hess = t2.batch_jacobian(jh, Xnew)
#     print(f"hess.shape: {hess.shape}")
#     hess = tf.transpose(hess, [2, 3, 0, 1])
#     hess = tf.linalg.diag_part(hess)
#     hess = tf.transpose(hess, [2, 0, 1])
#     print(f"hess.shape: {hess.shape}")
#     tf.print("hess")
#     tf.print(hess)

#     def hessian_rbf_cov_fn_wrt_single_x1x1(x1: InputData):
#         x1 = tf.expand_dims(x1, 1)
#         l2 = kernel.lengthscales ** 2
#         l2 = tf.linalg.diag(l2)
#         hessian = l2 * kernel.K(x1)
#         return hessian

#     hess = hessian_rbf_cov_fn_wrt_single_x1x1(Xnew)
#     print(f"hess.shape: {hess.shape}")

#     @tf.function()
#     def base_conditional_closure(args):
#         Kmn, Knn = args
#         jac_mean, jac_cov = base_conditional(
#             Kmn,
#             Kxx,
#             Knn,
#             f,
#             full_cov=True,
#             # full_cov=full_cov,
#             q_sqrt=q_sqrt,
#             white=whiten,
#         )
#         return jac_mean, jac_cov

#     jac_mean, jac_cov = tf.map_fn(
#         base_conditional_closure,
#         (jac, hess),
#         fn_output_signature=(default_float(), default_float()),
#     )
#     print("jac var")
#     print(jac_cov.shape)
#     print(jac_mean.shape)

#     # TODO mean function??
#     with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t3:
#         t3.watch(Xnew)
#         mu = mean_function(Xnew)
#     d_mu = t3.batch_jacobian(mu, Xnew)
#     d_mu = tf.transpose(d_mu, [0, 2, 1])
#     jac_mean += d_mu

#     return jac_mean, jac_cov


# @tf.function()
def gp_predict_jacobian(
    Xnew: ttf.Tensor2[NumData, InputDim],
    X: ttf.Tensor2[NumData, InputDim],
    # inducing_variable: InducingVariables,
    # X: InputData,
    kernel: Kernel,
    mean_function: MeanFunction,
    f,
    full_cov: Optional[bool] = False,
    q_sqrt=None,
    whiten: Optional[bool] = False,
) -> JacMeanAndVariance:
    def single_gp_predict_jacobian_closure(args):
        xnew = args
        return single_gp_predict_jacobian(
            xnew=tf.reshape(xnew, (1, -1)),
            X=X,
            kernel=kernel,
            mean_function=mean_function,
            f=f,
            full_cov=full_cov,
            q_sqrt=q_sqrt,
            whiten=whiten,
        )

    jac_mean, jac_cov = tf.map_fn(
        single_gp_predict_jacobian_closure,
        (Xnew),
        fn_output_signature=(default_float(), default_float()),
    )
    return jac_mean, jac_cov


# @tf.function
def single_gp_predict_jacobian(
    xnew: ttf.Tensor2[One, InputDim],
    X: ttf.Tensor2[NumData, InputDim],
    kernel: Kernel,
    mean_function: MeanFunction,
    f,
    full_cov: Optional[bool] = False,
    q_sqrt=None,
    whiten: Optional[bool] = False,
) -> JacMeanAndVariance:
    jitter = default_jitter()
    Kxx = kernel.K(X)
    print("Kxx")
    print(Kxx.shape)

    jac = grad_kernel_wrt_x1_via_distance(kernel, xnew, X)

    # hess_cov_fn = kernel.K(xnew)
    # hess_new = tf.hessians(hess_cov_fn, [xnew])[0][:, :, 0, :]
    hess = grad_kernel_wrt_x1x2_via_distance(kernel, xnew, xnew)

    # hess = grad_rbf_kernel_wrt_single_x1x1(kernel, xnew)
    # print(f"hess.shape: {hess.shape}")

    @tf.function
    def base_conditional_closure(args):
        Kmn, Knn = args
        jac_mean, jac_cov = base_conditional(
            Kmn,
            Kxx,
            Knn,
            f,
            full_cov=True,
            # full_cov=full_cov,
            q_sqrt=q_sqrt,
            white=whiten,
        )
        return jac_mean, jac_cov

    jac_mean, jac_cov = tf.map_fn(
        base_conditional_closure,
        (jac, hess),
        fn_output_signature=(default_float(), default_float()),
    )
    print("jac mean")
    print(jac_mean.shape)
    print(jac_cov.shape)

    dmu_dx = mean_function_jacobian(mean_function, xnew)
    jac_mean += dmu_dx

    try:
        tf.linalg.cholesky(jac_cov)
        # tf.print("cholesky PASSED")
    except:
        tf.print("cholesky FAILED")

    return jac_mean[0, :, :], jac_cov[0, :, :]


def mean_function_jacobian(mean_function, xnew):
    """Derivative of mean function wrt input xnew"""
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t_mu:
        t_mu.watch(xnew)
        mu = mean_function(xnew)
    dmu_dx = t_mu.batch_jacobian(mu, xnew)
    dmu_dx = tf.transpose(dmu_dx, [0, 2, 1])
    return dmu_dx


def grad_kernel_wrt_x1(kernel, x1, X2):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t_x1:
        t_x1.watch(x1)
        dk = kernel.K(x1, X2)
    jac = t_x1.batch_jacobian(dk, x1)
    print(f"jac.shape: {jac.shape}")
    return jac


def grad_kernel_wrt_x1_via_distance(kernel, x1, X2):
    """Kernel Jacobian dk_dx1 via partial derivatives of distance"""
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t_x1:
        t_x1.watch(x1)
        dist = kernel.scaled_squared_euclid_dist(x1, X2)
    ddist_dx = t_x1.batch_jacobian(dist, x1)

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t_dist:
        t_dist.watch(dist)
        k = kernel.K_r2(dist)
    dk_ddist = t_dist.batch_jacobian(k, dist)

    jac = tf.einsum("bik,bkm->bim", dk_ddist, ddist_dx)
    # print(f"jac.shape: {jac.shape}")
    return jac


def grad_kernel_wrt_x1x1_via_distance(kernel, x1):
    """Kernel Hessian d2k_dx1dx1 via partial derivatives of distance"""
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
    hess = hess[0, :, :, :]
    # print(f"hess.shape: {hess.shape}")
    return hess


def grad_rbf_kernel_wrt_single_x1x1(kernel: Kernel, x1: InputData):
    x1 = tf.expand_dims(x1, 1)
    l2 = kernel.lengthscales ** 2
    l2 = tf.linalg.diag(l2)
    hessian = l2 * kernel.K(x1)
    return hessian


# SVGP.predict_jacobian = predict_jacobian


# def predict_jacobian(
#     gp, Xnew: ttf.Tensor2[NumData, InputDim], full_cov: Optional[bool] = False
# ):
#     f_mean, f_var = gp.predict_f(Xnew, full_cov=full_cov)
#     print("f_mean")
#     print(f_mean.shape)
#     print(f_var.shape)
#     # mean = gp.mean_function(Xnew)
#     # print("mean.shape")
#     # print(mean.shape)
#     # f_mean = f_mean - mean

#     jac_mean, jac_cov = gp_predict_jacobian(
#         Xnew,
#         gp.inducing_variable,
#         gp.kernel,
#         gp.mean_function,
#         f=f_mean,
#         f_var=f_var,
#         q_mu=gp.q_mu,
#         full_cov=full_cov,
#         q_sqrt=gp.q_sqrt,
#         whiten=gp.whiten,
#     )
#     return jac_mean, jac_cov


# def gp_predict_jacobian(
#     Xnew: ttf.Tensor2[NumData, InputDim],
#     # X: ttf.Tensor2[NumData, InputDim],
#     inducing_variable: InducingVariables,
#     # X: InputData,
#     kernel: Kernel,
#     mean_function: MeanFunction,
#     f,
#     f_var,
#     q_mu,
#     full_cov: Optional[bool] = False,
#     q_sqrt=None,
#     whiten: Optional[bool] = False,
# ) -> JacMeanAndVariance:
#     # mu, var = _conditional(
#     #     Xnew,
#     #     inducing_variable.Z,
#     #     kernel,
#     #     f=f,
#     #     q_sqrt=q_sqrt,
#     #     full_cov=full_cov,
#     #     white=whiten,
#     #     # full_output_cov=full_output_cov,
#     # )
#     # mu = mu + mean_function(Xnew)
#     # print("f*")
#     # print(mu.shape)
#     # print(var.shape)

#     def single_gp_predict_jacobian_closure(args):
#         xnew, f_, f_var = args
#         return single_gp_predict_jacobian(
#             xnew=tf.reshape(xnew, (1, -1)),
#             inducing_variable=inducing_variable,
#             kernel=kernel,
#             mean_function=mean_function,
#             # f=f,
#             f=tf.reshape(f_, (1, -1)),
#             f_var=tf.reshape(f_var, (1, -1)),
#             q_mu=q_mu,
#             full_cov=full_cov,
#             q_sqrt=q_sqrt,
#             whiten=whiten,
#         )

#     jac_mean, jac_cov = tf.map_fn(
#         single_gp_predict_jacobian_closure,
#         (Xnew, f, f_var),
#         fn_output_signature=(default_float(), default_float()),
#     )
#     return jac_mean, jac_cov


# @tf.function()
# def single_gp_predict_jacobian(
#     xnew: ttf.Tensor2[One, InputDim],
#     # X: ttf.Tensor2[NumData, InputDim],
#     inducing_variable: InducingVariables,
#     # X: InputData,
#     kernel: Kernel,
#     mean_function: MeanFunction,
#     f,
#     f_var,
#     q_mu,
#     full_cov: Optional[bool] = False,
#     q_sqrt=None,
#     whiten: Optional[bool] = False,
# ) -> JacMeanAndVariance:

#     cov_fn = partial(
#         covariance_conditional,
#         kernel=kernel,
#         inducing_variable=inducing_variable,
#         f=q_mu,
#         q_sqrt=q_sqrt,
#         white=whiten,
#     )
#     print("xnew.shape")
#     print(xnew.shape)
#     Kxx = cov_fn(xnew, xnew)[0, :, :]
#     Kxx = f_var
#     print("Kxx")
#     print(Kxx)

#     def jac_kern_fn_wrapper(x1):
#         # x1 = x1.reshape(1, -1)
#         K = cov_fn(x1, xnew)
#         print("jac K")
#         print(K)
#         return K[0, :, :]

#     def hess_kern_fn_wrapper(x1, x2):
#         K = cov_fn(x1, x2)
#         print("hess K")
#         print(K)
#         return K[0, :, :]

#     def hessian_rbf_cov_fn_wrt_single_x1x1(x1: InputData):
#         x1 = tf.expand_dims(x1, 1)
#         l2 = kernel.lengthscales ** 2
#         l2 = tf.linalg.diag(l2)
#         hessian = l2 * kernel.K(x1)
#         return hessian

#     with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t2:
#         t2.watch(xnew)
#         with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t1:
#             t1.watch(xnew)
#             dk = jac_kern_fn_wrapper(xnew)
#             print("dk")
#             print(dk)
#             d2k = hess_kern_fn_wrapper(xnew, xnew)
#             print("d2k")
#             print(d2k)

#         jac = t1.batch_jacobian(dk, xnew)
#         jh = t1.batch_jacobian(d2k, xnew)
#     print(f"jac.shape: {jac.shape}")
#     hess = t2.batch_jacobian(jh, xnew)
#     hess = tf.transpose(hess, [2, 3, 0, 1])
#     hess = tf.linalg.diag_part(hess)
#     hess = tf.transpose(hess, [2, 0, 1])
#     # print("hess")
#     # print(hess)
#     # print("jac")
#     # print(jac)
#     # print(f)
#     # hess = hessian_rbf_cov_fn_wrt_single_x1x1(xnew)
#     # print(f"hess.shape: {hess.shape}")

#     hess_cov_fn = cov_fn(xnew, xnew)
#     hess = tf.hessians(hess_cov_fn, [xnew, xnew])[0][0, :, 0, :]
#     print("hess")
#     print(hess)
#     print(f"hess.shape: {hess.shape}")

#     f_sqrt = tf.linalg.cholesky(f_var)
#     jac_mean, jac_cov = base_conditional(
#         Kmn=jac,
#         Kmm=Kxx,
#         Knn=hess,
#         f=f,
#         full_cov=True,
#         # full_cov=full_cov,
#         # q_sqrt=None,
#         q_sqrt=f_sqrt,
#         # q_sqrt=q_sqrt,
#         # white=False,
#         white=whiten,
#     )
#     tf.print("jac_mean")
#     tf.print(jac_mean)
#     tf.print(jac_cov)

#     with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t3:
#         t3.watch(xnew)
#         mu = mean_function(xnew)
#     d_mu = t3.batch_jacobian(mu, xnew)
#     print("d_mu")
#     print(d_mu)
#     d_mu = tf.transpose(d_mu, [0, 2, 1])
#     print(d_mu)
#     # print(jac_cov)
#     jac_mean += d_mu
#     tf.print("d_mu")
#     tf.print(d_mu)
#     # print("d2k")
#     # print("jac var")
#     # print(jac_cov.shape)
#     # print(jac_mean.shape)
#     # TODO mean function??
#     return jac_mean[0, :, :], jac_cov[0, :, :, :]
