u"""
Created on 2016-5-17

@author: cheng.li
"""

import sys
from ctypes import *
import abc
from typing import Tuple
import numpy as np
from FactorModel.regulator import Constraints

if sys.platform == "win32":
    dll_handle = CDLL("optimizer.dll")
else:
    dll_handle = CDLL("liboptimizer.so")


class OptProblem(metaclass=abc.ABCMeta):

    def __init__(self,
                 cov: np.array,
                 er: np.array,
                 constraints: Constraints) -> None:
        self.cov = cov
        self.er = er
        self.constraints = constraints

    @abc.abstractmethod
    def optimize(self, cw: np.array) -> Tuple[np.array, float]:
        pass


class NoCostProblem(OptProblem):

    def __init__(self,
                 cov: np.array,
                 er: np.array,
                 constraints: Constraints) -> None:
        super().__init__(cov, er, constraints)

    def optimize(self, cw: np.array) -> Tuple[np.array, float]:
        return portfolio_optimizer_with_no_cost_penlty(self.cov,
                                                       self.er,
                                                       cw,
                                                       self.constraints.lb,
                                                       self.constraints.ub,
                                                       self.constraints.lc)


class CostBudgetProblem(OptProblem):

    def __init__(self,
                 cov: np.array,
                 er: np.array,
                 constraints: Constraints,
                 tc: np.array,
                 cost_budget: float) -> None:
        super().__init__(cov, er, constraints)
        self.tc = tc
        self.cost_budget = cost_budget

    def optimize(self, cw: np.array) -> Tuple[np.array, float]:
        return portfolio_optimizer_with_cost_budget(self.cov,
                                                    self.er,
                                                    self.tc,
                                                    cw,
                                                    self.cost_budget,
                                                    self.constraints.lb,
                                                    self.constraints.ub,
                                                    self.constraints.lc)


def set_stop_condition(epsg,
                       epsf,
                       epsx,
                       maxits):
    c_epsg = c_double(epsg)
    c_epsf = c_double(epsf)
    c_epsx = c_double(epsx)
    c_maxits = c_int(maxits)
    return dll_handle.setStopCondition(c_epsg,
                                       c_epsf,
                                       c_epsx,
                                       c_maxits)


def transform_pyarray_to_c_arr(input, add_value=0.0):
    try:
        _ = input.size
        flag = True
    except AttributeError:
        flag = False
    if flag:
        input = np.array(input) + add_value
        input.shape = 1, -1
        return (c_double * np.size(input))(*list(input.flat))
    else:
        return input


def argument_checker(cov,
                     er,
                     tc,
                     cw,
                     bndl,
                     bndu,
                     lc,
                     added=0.0):
    prob_size = len(er)

    c_er = transform_pyarray_to_c_arr(er)
    c_cov = transform_pyarray_to_c_arr(cov)
    c_tc = transform_pyarray_to_c_arr(tc, added)
    c_cw = transform_pyarray_to_c_arr(cw)

    c_bndl = transform_pyarray_to_c_arr(bndl)
    c_bndu = transform_pyarray_to_c_arr(bndu)
    c_lc = transform_pyarray_to_c_arr(lc)

    if not c_bndl:
        c_bndl = transform_pyarray_to_c_arr(-np.ones(prob_size) * 1e15)

    if not c_bndu:
        c_bndu = transform_pyarray_to_c_arr(np.ones(prob_size) * 1e15)

    if c_lc:
        c_lcm = int(len(c_lc) / (prob_size + 2))
    else:
        c_lcm = 0

    return prob_size, \
        c_er, \
        c_cov, \
        c_tc, \
        c_cw,  \
        c_bndl, \
        c_bndu, \
        c_lc, \
        c_lcm


def portfolio_optimizer_with_no_cost_penlty(cov,
                                            er,
                                            cw,
                                            bndl=None,
                                            bndu=None,
                                            lc=None):

    tc = np.array([0.0])
    prob_size, c_er, c_cov, _, c_cw, c_bndl, c_bndu, c_lc, c_lcm = \
        argument_checker(cov, er, tc, cw, bndl, bndu, lc)

    c_tw = (c_double * prob_size)(0., 0.)
    c_cost = (c_double * 1)(0.)

    dll_handle.portfolioOptimizerWithoutTradingCostPenalty(prob_size,
                                                           c_cov,
                                                           c_er,
                                                           c_cw,
                                                           c_bndl,
                                                           c_bndu,
                                                           c_lcm,
                                                           c_lc,
                                                           c_tw,
                                                           c_cost)

    target_weight = [0.0] * len(c_tw)
    for i in range(len(target_weight)):
        target_weight[i] = c_tw[i]
    cost = c_cost[0]
    return target_weight, cost


def portfolio_optimizer_with_cost_budget(cov,
                                        er,
                                        tc,
                                        cw,
                                        tcb,
                                        bndl,
                                        bndu,
                                        lc=None):
    prob_size, c_er, c_cov, c_tc, c_cw, c_bndl, c_bndu, c_lc, c_lcm = \
        argument_checker(cov, er, tc, cw, bndl, bndu, lc, 1e-15)

    c_tcb = c_double(tcb)
    c_tw = (c_double * prob_size)(0., 0.)
    c_cost = (c_double * 1)(0.)

    dll_handle.portfolioOptimizerWithTradingCostBudget(prob_size,
                                                       c_cov,
                                                       c_er,
                                                       c_tc,
                                                       c_cw,
                                                       c_tcb,
                                                       c_bndl,
                                                       c_bndu,
                                                       c_lcm,
                                                       c_lc,
                                                       c_tw,
                                                       c_cost)

    target_weight = [0.0] * len(c_tw)
    for i in range(len(target_weight)):
        target_weight[i] = c_tw[i]
    cost = c_cost[0]
    return target_weight, cost
