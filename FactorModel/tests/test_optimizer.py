# -*- coding: utf-8 -*-
u"""
Created on 2016-8-24

@author: cheng.li
"""

import unittest
import numpy as np
from FactorModel.regulator import Constraints
from FactorModel.optimizer import portfolio_optimizer


class TestOptimizer(unittest.TestCase):

    @staticmethod
    def analytic_solution(er, cov, aeq, beq):
        c_inv = np.linalg.inv(cov)
        v1 = np.linalg.inv(aeq @ c_inv @ aeq.T)
        return c_inv @ (er.T - aeq.T @ v1 @ aeq @ c_inv @ er.T + aeq.T @ v1 @ beq)

    def test_optimize_without_constraint(self):
        er = np.array([.05, .05])
        cov = np.array([[.25, .10], [.10, .15]])
        cw = np.array([.5, .5])
        tc = np.array([0.002, 0.002])
        constraints = Constraints(lb=None,
                                  ub=None,
                                  lc=None,
                                  lct=None)

        target_weight, cost \
            = portfolio_optimizer(cov, er, tc, cw, constraints, method='no_cost')

        benchmark_weight = np.linalg.solve(cov, er)
        self.assertTrue(np.all(np.isclose(target_weight, benchmark_weight)))

    def test_optimize_with_linear_constraints(self):
        er = np.array([.05, .05, .05])
        cov = np.array([[.25, .10, .08], [.10, .17, .05], [.08, .05, .15]])
        cw = np.array([0.33, 0.33, 0.33])
        tc = np.array([0.002, 0.002])

        lb = np.array([0., 0., 0.])
        ub = np.array([1., 1., 1.])

        lc = np.array([[1., 1., 1., 1.]])
        lct = np.array([0])

        constraints = Constraints(lb=lb,
                                  ub=ub,
                                  lc=lc,
                                  lct=lct)

        target_weight, cost \
            = portfolio_optimizer(cov, er, tc, cw, constraints, method='no_cost')

        aeq = lc[:, :-1]
        beq = lc[:, -1]

        benchmark_weight = TestOptimizer.analytic_solution(er, cov, aeq, beq)
        self.assertTrue(np.all(np.isclose(target_weight, benchmark_weight)))
