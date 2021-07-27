#!/usr/bin/env python3
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from typing import Optional
from gpflow.kernels import Kernel
from gpflow.models import SVGP
from gpflow.mean_functions import MeanFunction
from gpflow.types import MeanAndVariance
from gpflow.conditionals import base_conditional
from gpflow.conditionals.util import separate_independent_conditional_implementation

from geoflow.custom_types import InputDim, NumData
from gpflow.config import default_float, default_jitter

InputData = None


def predict_jacobian(
    gp, Xnew: ttf.Tensor2[NumData, InputDim], full_cov: Optional[bool] = False
):
    jac_mean, jac_cov = gp_predict_jacobian(
        Xnew,
        gp.inducing_variable.Z,
        gp.kernel,
        gp.mean_function,
        gp.q_mu,
        full_cov=full_cov,
        q_sqrt=gp.q_sqrt,
        whiten=gp.whiten,
    )
    return jac_mean, jac_cov


# SVGP.predict_jacobian = predict_jacobian


# def gp_predict_jacobian(
def gp_predict_jacobian(
    Xnew: ttf.Tensor2[NumData, InputDim],
    X: ttf.Tensor2[NumData, InputDim],
    # X: InputData,
    kernel: Kernel,
    mean_function: MeanFunction,
    f,
    full_cov: Optional[bool] = False,
    q_sqrt=None,
    whiten: Optional[bool] = False,
) -> MeanAndVariance:
    jitter = default_jitter()
    # print(jitter)
    Kxx = kernel.K(X)
    # Kxx = kernel.K(X, full_cov=False)
    # Kxx = kernel(X, full_cov=False)
    # Kxx += jitter * tf.eye(Kxx.shape[-1])
    # print("Kxx")
    # print(Kxx.shape)
    # Kxx = tf.expand_dims(Kxx, 0)
    # num_data = Xnew.shape[0]
    # Kxx = tf.tile(Kxx, [num_data, 1, 1])
    # print(Kxx.shape)

    def jac_kern_fn_wrapper(x1):
        # x1 = x1.reshape(1, -1)
        K = kernel.K(x1, X)
        return K

    def hess_kern_fn_wrapper(x1, x2):
        # K = kernel(x1, full_cov=False)
        K = kernel.K(x1, x2)
        # K = tf.linalg.diag_part(K)
        # print("K hessdig")
        # print(K.shape)
        # print(K)
        # return tf.reshape(K, [-1, 1])
        return K

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t2:
        t2.watch(Xnew)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as t1:
            # with tf.GradientTape(persistent=False, watch_accessed_variables=False) as t2:
            #     with tf.GradientTape(persistent=False, watch_accessed_variables=False) as t1:
            t1.watch(Xnew)
            dk = jac_kern_fn_wrapper(Xnew)
            # print("dk")
            # print(dk)
            d2k = hess_kern_fn_wrapper(Xnew, Xnew)
            # print("d2k")
            # print(d2k)
            # d2k = kernel.K(Xnew, Xnew)

        jac = t1.batch_jacobian(dk, Xnew)
        jh = t1.batch_jacobian(d2k, Xnew)
    # print(f"jac.shape: {jac.shape}")
    # print(jac)
    # hess = t2.batch_jacobian(jh, Xnew)[:, 0, :, :]
    hess = t2.batch_jacobian(jh, Xnew)
    hess = tf.transpose(hess, [2, 3, 0, 1])
    hess = tf.linalg.diag_part(hess)
    hess = tf.transpose(hess, [2, 0, 1])
    # print(f"hess.shape: {hess.shape}")
    # print(hess.shape)
    # print(hess)

    def hessian_rbf_cov_fn_wrt_single_x1x1(x1: InputData):
        x1 = x1.reshape(1, -1)
        # l2 = kernel.lengthscales.value ** 2
        # l2 = - kernel.lengthscales.value
        l2 = params["lengthscales"] ** 2
        l2 = jnp.diag(l2)
        # hessian = l2 * kernel.K(params, x1, x1)
        hessian = l2 * kernel.K(params, x1)
        return hessian

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
    # print("jac var")
    # print(jac_cov.shape)
    # print(jac_mean.shape)
    # TODO mean function??
    return jac_mean, jac_cov
