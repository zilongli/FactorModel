# -*- coding: utf-8 -*-
u"""
Created on 2016-8-24

@author: cheng.li
"""

import unittest
import numpy as np
from FactorModel.regulator import Constraints
from FactorModel.optimizer import NoCostProblem


class TestOptimizer(unittest.TestCase):

    @staticmethod
    def analytic_solution(er, cov, aeq, beq):
        c_inv = np.linalg.inv(cov)
        v1 = np.linalg.inv(aeq @ c_inv @ aeq.T)
        return c_inv \
            @ (er.T - aeq.T @ v1 @ aeq @ c_inv @ er.T + aeq.T @ v1 @ beq)

    def test_optimize_without_constraint(self):
        er = np.array([.05, .05])
        cov = np.array([[.25, .10], [.10, .15]])
        cw = np.array([.5, .5])
        constraints = Constraints(lb=None,
                                  ub=None,
                                  lc=None,
                                  suspend=None)

        prob = NoCostProblem(cov, er, constraints)
        target_weight, cost = prob.optimize(cw)

        benchmark_weight = np.linalg.solve(cov, er)
        self.assertTrue(np.all(np.isclose(target_weight, benchmark_weight)))

    def test_optimize_with_linear_constraints(self):
        er = np.array([.05, .05, .05])
        cov = np.array([[.25, .10, .08], [.10, .17, .05], [.08, .05, .15]])
        cw = np.array([0.33, 0.33, 0.33])

        lb = np.array([0., 0., 0.])
        ub = np.array([1., 1., 1.])

        lc = np.array([[1., 1., 1., 1., 1.]])

        constraints = Constraints(lb=lb,
                                  ub=ub,
                                  lc=lc,
                                  suspend=None)

        prob = NoCostProblem(cov, er, constraints)
        target_weight, cost = prob.optimize(cw)

        aeq = np.array([[1., 1., 1.]])
        beq = np.array([1.])

        benchmark_weight = TestOptimizer.analytic_solution(er, cov, aeq, beq)
        self.assertTrue(np.all(np.isclose(target_weight, benchmark_weight)))

    def test_optimizer_with_multiple_constraints(self):
        er = np.array([.05, -.03, .02])
        cov = np.array([[.25, .10, .08], [.10, .17, .05], [.08, .05, .15]])
        cw = np.array([0.33, 0.33, 0.33])

        lb = np.array([0., 0., 0.])
        ub = np.array([1., 1., 1.])

        lc = np.array([[1., 1., 1., 1., 1.],
                       [1., 1., 0., .3, .6]])

        constraints = Constraints(lb=lb,
                                  ub=ub,
                                  lc=lc,
                                  suspend=None)

        prob = NoCostProblem(cov, er, constraints)
        target_weight, cost = prob.optimize(cw)

        expected_result = [0.4165, 0.0003, 0.5832]

        self.assertTrue(
            np.all(np.isclose(target_weight, expected_result, atol=1e-3)))
