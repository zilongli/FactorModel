# -*- coding: utf-8 -*-
u"""
Created on 2016-8-23

@author: cheng.li
"""

import unittest
import numpy as np
from FactorModel.regulator import Constraints
from FactorModel.optimizer import portfolio_optimizer


class TestOptimizer(unittest.TestCase):
    def test_optimize_without_constraints(self):
        er = [.05, .05]
        cov = [[.25, .10], [.10, .15]]
        cw = [.5, .5]
        tc = [0., 0.]
        constrants = Constraints(lb=None,
                                 ub=None,
                                 lc=None,
                                 lct=None)

        target_weight, cost \
            = portfolio_optimizer(cov, er, tc, cw, constrants, method='no_cost')


if __name__ == "__main__":
    pass
