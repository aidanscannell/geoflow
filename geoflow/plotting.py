#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from gpflow import default_float
from geoflow.gp import predict_jacobian
from geoflow.manifolds import GPManifold

cmap = cm.coolwarm
cmap = cm.PRGn
cmap = cm.PiYG


class ManifoldPlotter:
    def __init__(self, manifold: GPManifold, test_inputs=None):
        self.manifold = manifold
        if test_inputs is None:
            # Xnew, xx, yy = create_grid(svgp.inducing_variable, 961)
            N = 100
            # N = 1000
            sqrtN = int(np.sqrt(N))
            x1_low = -3
            x1_high = 3
            x2_low = -3
            x2_high = 3
            xx = np.linspace(x1_low, x1_high, sqrtN)
            yy = np.linspace(x2_low, x2_high, sqrtN)
            xx, yy = np.meshgrid(xx, yy)
            test_inputs = tf.constant(
                np.column_stack([xx.reshape(-1), yy.reshape(-1)]), dtype=default_float()
            )
        self.test_inputs = test_inputs
        self.figsize = (12, 4)

        self.gp = self.manifold.gp
        self.f_mean, self.f_var = self.gp.predict_f(self.test_inputs, full_cov=False)
        self.jac_mean, self.jac_var = predict_jacobian(
            self.gp, self.test_inputs, full_cov=False
        )
        print("self.jac_var")
        print(self.jac_var.shape)
        self.metric = self.manifold.metric(self.test_inputs)
        print("self.metric")
        print(self.metric.shape)
        self.metric_trace = tf.reshape(tf.linalg.trace(self.metric), (-1, 1))
        print("self.metric_trace")
        print(self.metric_trace.shape)

    def plot_jacobian_mean(self):
        fig, axs = self.create_jacobian_mean_fig_axs()
        self.plot_jacobian_mean_given_fig_axs(fig, axs)

    def plot_jacobian_var(self):
        fig, axs = self.create_jacobian_var_fig_axs()
        self.plot_jacobian_var_given_fig_axs(fig, axs)

    def plot_metric_trace(self):
        fig, axs = self.create_metric_trace_fig_axs()
        self.plot_metric_trace_given_fig_axs(fig, axs)

    def create_jacobian_mean_fig_axs(self):
        fig, axs = plt.subplots(1, 2, figsize=self.figsize)
        return fig, axs

    def create_jacobian_var_fig_axs(self):
        fig, axs = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1] * 2))
        plt.subplots_adjust(wspace=0, hspace=0)
        return fig, axs

    def create_metric_trace_fig_axs(self):
        # fig, axs = plt.subplots(1, 1, figsize=self.figsize)
        fig, axs = plt.subplots(1, 1, figsize=(self.figsize[0] / 2, self.figsize[1]))
        return fig, axs

    def plot_contourf(self, fig, ax, z, label="", levels=20, cbar=True):
        contf = ax.tricontourf(
            self.test_inputs[:, 0],
            self.test_inputs[:, 1],
            z[:, 0],
            levels=levels,
            cmap=cmap,
        )
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        if cbar:
            cbar = fig.colorbar(contf, shrink=0.5, aspect=5, ax=ax)
            cbar.set_label(label)

    def plot_mean_and_var(self, fig, axs, mean, var, llabel="Mean", rlabel="Variance"):
        self.plot_contourf(fig, axs[0], mean, label=llabel)
        self.plot_contourf(fig, axs[1], var, label=rlabel)

    def plot_jacobian_mean_given_fig_axs(self, fig, axs):
        self.plot_mean_and_var(fig, axs, self.f_mean, self.f_var)
        for ax in axs:
            ax.quiver(
                self.test_inputs[:, 0],
                self.test_inputs[:, 1],
                self.jac_mean[:, 0],
                self.jac_mean[:, 1],
            )

    def plot_jacobian_var_given_fig_axs(self, fig, axs):
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                self.plot_contourf(fig, axs[j, i], self.jac_var[:, :, j, i])

    def plot_metric_trace_given_fig_axs(self, fig, axs):
        self.plot_contourf(fig, axs, self.metric_trace)


def plot_svgp_jacobian_mean(svgp, test_inputs=None):
    if test_inputs == None:
        # Xnew, xx, yy = create_grid(svgp.inducing_variable, 961)
        N = 100
        # N = 1000
        sqrtN = int(np.sqrt(N))
        x1_low = -3
        x1_high = 3
        x2_low = -3
        x2_high = 3
        xx = np.linspace(x1_low, x1_high, sqrtN)
        yy = np.linspace(x2_low, x2_high, sqrtN)
        xx, yy = np.meshgrid(xx, yy)
        test_inputs = tf.constant(
            np.column_stack([xx.reshape(-1), yy.reshape(-1)]), dtype=default_float()
        )

    fmean, fvar = svgp.predict_f(test_inputs, full_cov=False)
