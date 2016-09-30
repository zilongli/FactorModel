# -*- coding: utf-8 -*-
u"""
Created on 2016-9-30

@author: cheng.li
"""

import numpy as np


class VectorAR1Process(object):

    def __init__(self, initial, phi, cov):
        self.initial = np.array(initial)
        self.phi = phi
        self.n = len(self.initial)

        self.expect = self.initial
        self.trajectory = []
        self.cholesky = np.linalg.cholesky(cov).T

    def evolve(self):
        diffusion = np.random.randn(self.n) @ self.cholesky
        self.expect = self.phi * self.expect + diffusion
        self._push(self.expect)

    def _push(self, new_sample):
        self.trajectory.append(new_sample)


if __name__ == "__main__":
    sigma = 0.02
    num_assets = 500
    processes = VectorAR1Process(np.zeros(num_assets),
                                 0.25,
                                 (0.02 ** 2) * np.diag(np.ones(num_assets)))
    for i in range(50000):
        processes.evolve()

    data = np.array(processes.trajectory)

    np.corrcoef(data.T)

    np.corrcoef(data[:-1, 0], data[1:, 0])
