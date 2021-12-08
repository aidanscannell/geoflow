#!/usr/bin/env python3
from typing import Optional, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from geoflow.custom_types import (
    InputDim,
    JacMeanAndVariance,
    NumData,
    NumInducing,
    One,
    OutputDim,
)
from geoflow.gp.covariances import hess_Kuu_wrt_Xnew, jac_Kuf_wrt_Xnew
from gpflow import posteriors
from gpflow.conditionals import base_conditional
from gpflow.kernels import Kernel, SquaredExponential
from gpflow.mean_functions import MeanFunction
from gpflow.utilities.ops import leading_transpose


def jacobian_conditional_with_precompute(
    Xnew,
    inducing_variable,
    kernel: Kernel,
    mean_function: MeanFunction,
    alpha: ttf.Tensor2[NumInducing, OutputDim],
    Qinv: ttf.Tensor2[NumInducing, OutputDim],
    num_latent_gps: int = 1,
    full_cov: bool = True,
    full_output_cov: bool = False,
) -> JacMeanAndVariance:
    """Computes predictive mean and (co)variance of derivative wrt Xnew"""
    Knn = hess_Kuu_wrt_Xnew(
        kernel, Xnew, inducing_variable, full_cov=full_cov
    )  # [N, F, D, D]
    Kus = jac_Kuf_wrt_Xnew(kernel, Xnew, inducing_variable)  # [N, F, M, D]
    # print(f"jac.shape: {Kus.shape}")
    # print(f"hess.shape: {Knn.shape}")

    Ksu = leading_transpose(Kus, [..., -1, -2])
    jac_mean = Ksu @ alpha
    jac_mean = leading_transpose(jac_mean[..., 0], [..., -1, -2])  # [N, D, F]

    if full_cov:
        jac_cov = Knn - Ksu @ Qinv @ Kus  # [N, F, D, D]
    else:
        # TODO haven't checked this
        Kfu_Qinv_Kuf = tf.reduce_sum(Kus * tf.matmul(Qinv, Kus), axis=-2)
        jac_cov = Knn - Kfu_Qinv_Kuf
        # jac_cov = tf.tile(var[:, None], [1, num_latent_gps])

    # Check covariance matrix is positive semi-definite
    # try:
    #     tf.linalg.cholesky(jac_cov)
    # except:
    #     tf.print("cholesky FAILED")

    # Add derivate of mean_function to jac_mean
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(Xnew)
        mean = mean_function(Xnew)
    dmean_dx = tape.batch_jacobian(mean, Xnew)
    dmean_dx = tf.transpose(dmean_dx, [0, 2, 1])
    jac_mean = jac_mean + dmean_dx

    return jac_mean, jac_cov


def jacobian_conditional(
    Xnew: ttf.Tensor2[NumData, InputDim],
    X: Union[ttf.Tensor2[NumData, InputDim], ttf.Tensor2[NumInducing, InputDim]],
    kernel: Kernel,
    mean_function: MeanFunction,
    f: Union[ttf.Tensor2[NumInducing, OutputDim], ttf.Tensor2[NumData, OutputDim]],
    full_cov: Optional[bool] = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[
        Union[
            ttf.Tensor2[NumInducing, OutputDim],
            ttf.Tensor3[OutputDim, NumInducing, NumInducing],
        ]
    ] = None,
    whiten: Optional[bool] = False,
    precompute_cache: Optional[
        posteriors.PrecomputeCacheType
    ] = posteriors.PrecomputeCacheType.TENSOR,
) -> JacMeanAndVariance:
    """Mean and covariance of GP derivative at xnew for single-output GP"""
    # TODO what's the best way to get num_latent_gps
    num_latent_gps = f.shape[-1]
    print("hi aidan")
    posterior = posteriors.create_posterior(
        kernel=kernel,
        inducing_variable=X,
        q_mu=f,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute_cache=precompute_cache,
    )
    jac_mean, jac_cov = jacobian_conditional_with_precompute(
        Xnew,
        posterior.X_data,
        kernel=posterior.kernel,
        mean_function=posterior.mean_function,
        alpha=posterior.alpha,
        Qinv=posterior.Qinv,
        num_latent_gps=num_latent_gps,
        full_cov=full_cov,
        full_output_cov=full_output_cov,
    )
    return jac_mean, jac_cov
