u"""
Created on 2016-5-17

@author: cheng.li
"""

import os
import sys
from ctypes import *
import numpy as np

dir_name = os.path.dirname(__file__)

if sys.platform == "win32":
    alglib_dll_path = os.path.join(dir_name, 'lib/alglib.dll')
    optimizer_dll_path = os.path.join(dir_name, 'lib/optimizer.dll')
else:
    alglib_dll_path = os.path.join(dir_name, 'lib/libalglib.so')
    optimizer_dll_path = os.path.join(dir_name, 'lib/liboptimizer.so')

_ = CDLL(alglib_dll_path)
dll_handle = CDLL(optimizer_dll_path)


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
                     lct,
                     added=0.0):
    prob_size = len(er)

    c_er = transform_pyarray_to_c_arr(er)
    c_cov = transform_pyarray_to_c_arr(cov)
    c_tc = transform_pyarray_to_c_arr(tc, added)
    c_cw = transform_pyarray_to_c_arr(cw)

    c_bndl = transform_pyarray_to_c_arr(bndl)
    c_bndu = transform_pyarray_to_c_arr(bndu)
    c_lc = transform_pyarray_to_c_arr(lc)
    c_lct = transform_pyarray_to_c_arr(lct)
    if c_lct:
        c_lcm = len(c_lct)
    else:
        c_lcm = 0

    return prob_size, c_er, c_cov, c_tc, c_cw, c_bndl, c_bndu, c_lc, c_lct, c_lcm


def portfolio_optimizer_with_no_cost_penlty(cov,
                                           er,
                                           cw,
                                           bndl=None,
                                           bndu=None,
                                           lc=None,
                                           lct=None):

    tc = np.array([0.0])
    prob_size, c_er, c_cov, _, c_cw, c_bndl, c_bndu, c_lc, c_lct, c_lcm = \
        argument_checker(cov, er, tc, cw, bndl, bndu, lc, lct)

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
                                  c_lct,
                                  c_tw,
                                  c_cost)

    target_weight = [0.0] * len(c_tw)
    for i in range(len(target_weight)):
        target_weight[i] = c_tw[i]
    cost = c_cost[0]
    return target_weight, cost


def portfolio_optimizer_with_no_cost_penlty2(cov,
                                                er,
                                                cw,
                                                bndl=None,
                                                bndu=None,
                                                lc=None,
                                                lct=None):
    np.array([0.0])
    prob_size, c_er, c_cov, _, c_cw, c_bndl, c_bndu, c_lc, c_lct, c_lcm = \
        argument_checker(cov, er, tc, cw, bndl, bndu, lc, lct)

    c_tw = (c_double * prob_size)(0., 0.)
    c_cost = (c_double * 1)(0.)

    dll_handle.portfolioOptimizerWithoutTradingCostPenalty2(prob_size,
                                                           c_cov,
                                                           c_er,
                                                           c_cw,
                                                           c_bndl,
                                                           c_bndu,
                                                           c_lcm,
                                                           c_lc,
                                                           c_lct,
                                                           c_tw,
                                                           c_cost)

    target_weight = [0.0] * len(c_tw)
    for i in range(len(target_weight)):
        target_weight[i] = c_tw[i]
    cost = c_cost[0]
    return target_weight, cost


def portfolio_optimizer_with_cost_buget(cov,
                                        er,
                                        tc,
                                        cw,
                                        tcb,
                                        bndl,
                                        bndu,
                                        lc=None,
                                        lct=None):
    prob_size, c_er, c_cov, c_tc, c_cw, c_bndl, c_bndu, c_lc, c_lct, c_lcm = \
        argument_checker(cov, er, tc, cw, bndl, bndu, lc, lct, 1e-15)

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
                                                       c_lct,
                                                       c_tw,
                                                       c_cost)

    target_weight = [0.0] * len(c_tw)
    for i in range(len(target_weight)):
        target_weight[i] = c_tw[i]
    cost = c_cost[0]
    return target_weight, cost


def portfolio_optimizer(cov, er, tc, cw, constraints, method='no_cost', cost_buget=9999):
    if method == 'cost_budget':
        return portfolio_optimizer_with_cost_buget(cov,
                                                   er,
                                                   tc,
                                                   cw,
                                                   cost_buget,
                                                   constraints.lb,
                                                   constraints.ub,
                                                   constraints.lc,
                                                   constraints.lct)
    elif method == 'no_cost':
        return portfolio_optimizer_with_no_cost_penlty(cov,
                                                      er,
                                                      cw,
                                                      constraints.lb,
                                                      constraints.ub,
                                                      constraints.lc,
                                                      constraints.lct)

    elif method == 'no_cost2':
        return portfolio_optimizer_with_no_cost_penlty2(cov,
                                                       er,
                                                       cw,
                                                       constraints.lb,
                                                       constraints.ub,
                                                       constraints.lc,
                                                       constraints.lct)

    else:
        raise ValueError("({method}) is not recognized".fromat(method=method))


if __name__ == "__main__":

    from FactorModel.regulator import Constraints

    df = 1000

    er = np.random.randn(df) * 0.02
    cov = np.diag(np.abs(np.random.randn(df)) * 0.0004)
    tc = np.ones(df) * 0.003
    cw = np.ones(df) / df
    lb = np.zeros(df)
    ub = np.ones(df)

    lc = np.ones(df + 1)
    lct = np.array([0.0])

    # ndim = 200
    # cov = np.diag(np.ones(ndim))
    # er = np.random.randn(ndim)
    # tc = np.zeros(ndim)
    # cw = np.zeros(ndim)
    #
    # lb = -np.ones(ndim)
    # ub = np.ones(ndim)

    # lc = np.array([1.0, 1.0, 1.0])
    # lct = np.array([0.0])
    constraints = Constraints(lb, ub, lc, lct)

    cond = set_stop_condition(1e-8, 1e-8, 1e-8, 30000)
    import datetime as dt

    start = dt.datetime.now()
    reps = 1
    for i in range(reps):
        cond = portfolio_optimizer(cov, er, tc, cw, constraints, method='no_cost')
    end = dt.datetime.now()
    print(end - start)

    start = dt.datetime.now()
    for i in range(reps):
        cond = portfolio_optimizer(cov, er, tc, cw, constraints, method='no_cost2')
    end = dt.datetime.now()
    print(end - start)

    start = dt.datetime.now()
    for i in range(reps):
        cond = portfolio_optimizer(cov, er, tc, cw, constraints, method='cost_budget', cost_buget=9999)#0.0003)
    end = dt.datetime.now()
    print(end - start)
