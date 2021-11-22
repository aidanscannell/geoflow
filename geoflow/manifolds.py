#!/usr/bin/env python3
import abc
import typing
from typing import Optional, Union

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow.models import GPModel
from gpflow.types import MeanAndVariance
from tensor_annotations import axes

from geoflow.custom_types import InputDim, NumData
from geoflow.gp import predict_jacobian

TwoInputDim = typing.NewType("TwoInputDim", axes.Axis)  # 2*InputDim


class Manifold(abc.ABC):
    def length(self, curve):
        """Compute the discrete length of a given curve."""
        raise NotImplementedError

    def energy(self, positions, velocities):
        """Computes the discrete energy of a trajectory.

        :param trajectory:
        :returns: energy of the curve
        """
        energy = self.inner_product(positions, velocities, velocities)
        print("energy yo")
        print(energy)
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
        vec_metric = lambda x: tf.reshape(self.metric(x), -1)
        grad_vec_metric_wrt_pos = jax.jacfwd(vec_metric)(pos)
        metric = self.metric(pos)
        inner_prod = self._inner_product_given_metric(metric, vel, vel)

        vel = vel.reshape(1, -1)
        kron_vel = tf.kron(vel, vel).T

        jitter = 1e-6
        pos_dim = pos.shape[0]
        metric += tf.eye(pos_dim) * jitter
        # rhs = grad_vec_metric_wrt_pos.T @ kron_vel
        # chol = jsp.linalg.cholesky(metric, lower=True)
        # inv_metric = jsp.linalg.solve_triangular(chol, rhs, lower=True)
        # print(inv_metric)
        # acc = -0.5 * inv_metric
        # print(acc)

        inv_metric = tf.linalg.inv(metric)
        acc = -0.5 * inv_metric @ grad_vec_metric_wrt_pos.T @ kron_vel

        state_prime = tf.concatenate([vel.T, acc], 0)
        return state_prime.flatten()

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
        return self._geodesic_ode(pos, vel)


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

    def metric(
        self, Xnew: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]]
    ) -> Union[
        ttf.Tensor2[InputDim, InputDim], ttf.Tensor3[NumData, InputDim, InputDim]
    ]:
        """Return the metric tensor at a specified point."""
        num_data = Xnew.shape[0]
        input_dim = Xnew.shape[1]
        jac_mean, jac_cov = self.embed_jac(Xnew)
        # print("jac_mean {}, jac_cov {}".format(jac_mean.shape, jac_cov.shape))
        print("jac")
        print(jac_mean)
        print(jac_cov)
        output_dim = jac_mean.shape[-1]
        assert jac_mean.shape == (num_data, input_dim, output_dim)
        assert jac_cov.shape == (num_data, output_dim, input_dim, input_dim)
        jac_prod = jac_mean @ tf.transpose(jac_mean, [0, 2, 1])
        assert jac_prod.shape == (num_data, input_dim, input_dim)
        # metric = jac_prod + self.covariance_weight * jac_cov
        metric = jac_prod + self.covariance_weight * tf.reduce_sum(jac_cov, 1)
        assert metric.shape == (num_data, input_dim, input_dim)
        return metric

    def embed(
        self, Xnew: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]]
    ) -> MeanAndVariance:
        """Embed Xnew into (mu, var) space."""
        means, vars = self.gp.predict_f(Xnew, full_cov=False)
        return means, vars
        # return means.reshape(-1), vars.reshape(-1)

    def embed_jac(
        self, Xnew: Union[ttf.Tensor1[InputDim], ttf.Tensor2[NumData, InputDim]]
    ):
        """Embed the manifold into (mu, var) space."""
        # jac_mean, jac_cov = self.gp.predict_jacobian(Xnew)
        jac_mean, jac_cov = predict_jacobian(self.gp, Xnew)
        # jac_mean, jac_cov = gp_predict_jacobian(Xnew)
        return jac_mean, jac_cov
