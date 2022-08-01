#!/usr/bin/env python3
import abc
import typing
from typing import Optional, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float, posteriors
from gpflow.utilities.ops import leading_transpose
from gpflow.models import GPModel
from tensor_annotations import axes

from geoflow.custom_types import InputDim, JacMeanAndVariance, MeanAndVariance, NumData
from geoflow.gp.conditionals import jacobian_conditional_with_precompute

TwoInputDim = typing.NewType("TwoInputDim", axes.Axis)  # 2*InputDim


class Manifold(abc.ABC):

    # JITTER = 1e-6

    def length(self, curve):
        """Compute the discrete length of a given curve."""
        raise NotImplementedError

    def energy(self, positions, velocities):
        """Computes the discrete energy of a trajectory.

        :param trajectory:
        :returns: energy of the curve
        """
        energy = self.inner_product(positions, velocities, velocities)
        return tf.reduce_sum(energy)

    def metric(
        self, points: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]]
    ) -> Union[
        ttf.Tensor2[InputDim, InputDim], ttf.Tensor3[NumData, InputDim, InputDim]
    ]:
        """Return the metric tensor at a specified SET of points."""
        # return self._metric(points)
        raise NotImplementedError

    def inner_product_given_metric(self, metric, U, V):
        """Compute inner product between sets of tangent vectors u and v at base points.

        :param metric: metric tensor at base points [num_data, input_dim, input_dim]
        :param U: set of tangent vectors in tangent space of base [num_data, input_dim]
        :param V: set of tangent vectors in tangent space of base [num_data, input_dim]
        :returns: inner product between u and v according to metric @ base [num_data]
        """
        assert len(metric.shape) == 3
        assert len(U.shape) == len(V.shape)
        if len(U.shape) == 2:
            U = tf.expand_dims(U, 1)
            V = tf.expand_dims(V, 1)
        assert len(U.shape) == 3
        return U @ metric @ tf.transpose(V, [0, 2, 1])
        # return self._inner_product_given_metric(metric, U, V)

    def inner_product(self, base, U, V):
        """Compute inner product between sets of tangent vectors u and v at base points.

        :param base: base points [num_data, input_dim, input_dim]
        :param U: set of tangent vectors in tangent space of base [num_data, input_dim]
        :param V: set of tangent vectors in tangent space of base [num_data, input_dim]
        :returns: inner product between u and v according to metric @ base [num_data]
        """
        metric = self.metric(base)
        return self.inner_product_given_metric(metric, U, V)

    def _geodesic_ode(
        self, pos: ttf.Tensor1[InputDim], vel: ttf.Tensor1[InputDim]
    ) -> ttf.Tensor1[TwoInputDim]:
        """Evaluate the geodesic ODE of the manifold."""
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(pos)
            metric = self.metric(pos)  # [..., D, D]
            print("metric.shape")
            print(metric.shape)
            vec_metric = tf.reshape(metric, [*metric.shape[:-2], -1])  # [..., 2D]
        grad_vec_metric_wrt_pos = tape.batch_jacobian(vec_metric, pos)  # [..., 2D, D]
        print("grad_vec_metric_wrt_pos")
        print(grad_vec_metric_wrt_pos.shape)

        # grad_vec_metric_wrt_pos = jax.jacfwd(vec_metric)(pos)
        # metric = self.metric(pos)
        # inner_prod = self._inner_product_given_metric(metric, vel, vel)
        print("vel")
        print(vel.shape)
        if len(vel.shape) >= 2:
            vel_expanded = tf.expand_dims(vel, -2)
        operator_1 = tf.linalg.LinearOperatorFullMatrix(vel_expanded)
        # operator_2 = tf.linalg.LinearOperatorFullMatrix(
        #     leading_transpose(vel, [..., -1, -2])
        # )
        kron_operator = tf.linalg.LinearOperatorKronecker([operator_1, operator_1])
        kron_vel = kron_operator.to_dense()  # [..., D^2, 1]
        print("kron_vel")
        print(kron_vel.shape)

        # metric = metric + (
        #     tf.eye(
        #         metric.shape[-2],
        #         metric.shape[-1],
        #         batch_shape=metric.shape[:-2],
        #         dtype=default_float(),
        #     )
        #     # * self.JITTER
        # )
        # trans = leading_transpose(grad_vec_metric_wrt_pos, [..., -1, -2])
        # print("trans")
        # print(trans.shape)
        rhs = leading_transpose(kron_vel @ grad_vec_metric_wrt_pos, [..., -1, -2])
        print("rhs")
        print(rhs.shape)
        chol = tf.linalg.cholesky(metric)
        print("chol")
        print(chol.shape)
        acc = -0.5 * tf.linalg.cholesky_solve(chol, rhs)
        # inv_metric = jsp.linalg.solve_triangular(chol, rhs, lower=True)
        # print(inv_metric)
        # acc = -0.5 * inv_metric
        # print(acc)

        # inv_metric = tf.linalg.inv(metric)
        # print("inv_metric")
        # print(inv_metric.shape)
        # acc = -0.5 * inv_metric @ grad_vec_metric_wrt_pos @ kron_vel
        # acc = -0.5 * kron_vel @ grad_vec_metric_wrt_pos @ inv_metric
        print("acc.shape")
        print(acc.shape)
        state_prime = tf.concat([vel, acc[..., 0]], -1)
        print("state_prime")
        print(state_prime)
        # return state_prime[:, 0, :]
        return state_prime

        # state_prime = tf.concatenate([vel.T, acc], 0)
        # print("state_prime.shape")
        # print(state_prime.shape)
        # return state_prime.flatten()

    def geodesic_ode(
        self,
        pos: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]],
        vel: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]],
    ) -> Union[ttf.Tensor1[TwoInputDim], ttf.Tensor2[NumData, TwoInputDim]]:
        """Evaluate the geodesic ODE of the manifold.

        :param pos: array of points in the input space
        :param vel: array representing the velocities at the points
        :returns: array of accelerations at the points
        """
        print("geodeic_oded")
        print(pos.shape)
        print(vel.shape)
        print(tf.rank(pos))
        return self._geodesic_ode(pos, vel)
        # if len(pos.shape) == 1:
        #     return self._geodesic_ode(pos, vel)
        # else:
        #     print("MAPPIng")
        #     a = tf.map_fn(self._geodesic_ode, (pos, vel))
        #     print("a")
        #     print(a)
        #     return a


class GPManifold(Manifold):
    """
    A common interface for embedded manifolds. Specific embedded manifolds
    should inherit from this abstract base class abstraction.
    """

    def __init__(
        self,
        gp: GPModel,
        covariance_weight: Optional[float] = 1.0,
    ):
        self.gp = gp
        self.covariance_weight = covariance_weight
        self.posterior = gp.posterior(
            precompute_cache=posteriors.PrecomputeCacheType.TENSOR
        )
        self.num_latent_gps = gp.num_latent_gps

    def metric(
        self, Xnew: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]]
    ) -> Union[
        ttf.Tensor2[InputDim, InputDim], ttf.Tensor3[NumData, InputDim, InputDim]
    ]:
        """Return the metric tensor at a specified point."""
        num_data, input_dim = Xnew.shape
        jac_mean, jac_cov = self.embed_jac(Xnew, full_cov=True, full_output_cov=False)
        print("jac_mean {}, jac_cov {}".format(jac_mean.shape, jac_cov.shape))
        assert jac_mean.shape == (num_data, input_dim, self.num_latent_gps)
        assert jac_cov.shape == (num_data, self.num_latent_gps, input_dim, input_dim)
        jac_prod = jac_mean @ tf.transpose(jac_mean, [0, 2, 1])
        assert jac_prod.shape == (num_data, input_dim, input_dim)
        metric = jac_prod + self.covariance_weight * tf.reduce_prod(jac_cov, 1)
        assert metric.shape == (num_data, input_dim, input_dim)
        return metric

    def embed(
        self,
        Xnew: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]],
        full_cov: Optional[bool] = True,
        full_output_cov: Optional[bool] = False,
    ) -> MeanAndVariance:
        """Embed Xnew into (mu, var) space."""
        means, vars = self.posterior.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return means, vars

    def embed_jac(
        self,
        Xnew: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]],
        full_cov: Optional[bool] = True,
        full_output_cov: Optional[bool] = False,
    ) -> JacMeanAndVariance:
        """Embed the manifold into (mu, var) space."""
        print("Xnew.shape")
        print(Xnew.shape)
        print(self.posterior.X_data.Z.shape)
        print(self.posterior.cache[0].shape)
        print(self.posterior.cache[1].shape)
        jac_mean, jac_cov = jacobian_conditional_with_precompute(
            Xnew,
            self.posterior.X_data,
            kernel=self.posterior.kernel,
            mean_function=self.posterior.mean_function,
            alpha=self.posterior.cache[0],
            Qinv=self.posterior.cache[1],
            num_latent_gps=self.num_latent_gps,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
        )
        return jac_mean, jac_cov
