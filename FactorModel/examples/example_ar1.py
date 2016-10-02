# -*- coding: utf-8 -*-
u"""
Created on 2016-9-30

@author: cheng.li
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class VectorAR1Process(object):

    def __init__(self, initial, phi, cov):
        self.initial = np.array(initial)
        self.phi = phi
        self.n = len(self.initial)

        self.start_point = self.initial
        self.end_point = None
        self.trajectory = []
        self.cholesky = np.linalg.cholesky(cov).T

    def evolve(self, nsteps):
        for i in range(nsteps):
            diffusion = np.random.randn(self.n) @ self.cholesky
            self.end_point = self.phi * self.start_point + diffusion
            self._push(self.start_point)
            self.start_point = self.end_point

        self._push(self.end_point)

    def _push(self, new_sample):
        self.trajectory.append(new_sample)

    @property
    def samples(self):
        return np.array(self.trajectory)

    @property
    def size(self):
        return self.n

    def expectaton(self, begin):
        return self.phi * begin


class TradingPlan(object):

    def __init__(self, process, in_threshold, out_threshold):
        self.process = process
        self.in_threshold = in_threshold
        self.out_threshold = out_threshold
        self.target_num = self.in_threshold
        self.current_weight = np.zeros(self.process.size)

        self.steps = []
        self.codes = []
        self.ers = []
        self.real_rs = []
        self.previous_positions = []
        self.current_positions = []

    def simulate(self):
        trajectory = self.process.samples
        nsteps = np.size(trajectory, 0)
        for i in range(1, nsteps):
            begin_positions = self.current_weight.copy()
            start_point = trajectory[i-1, :]
            er = self.process.expectaton(start_point)
            real_r = trajectory[i, :]

            to_sort = -er
            order = to_sort.argsort()
            ranks = order.argsort()

            to_buy_candidates = (ranks < self.in_threshold) & (begin_positions == 0.)
            to_sell_candidates = (ranks >= self.out_threshold-1) & (begin_positions != 0.)
            to_keep = (ranks < self.out_threshold-1) & (begin_positions != 0)

            to_buy_in = min(self.target_num - np.sum(to_keep), np.sum(to_buy_candidates))

            filtered = to_sort.copy()
            filtered[~to_buy_candidates] = np.inf
            filtered_order = filtered.argsort()

            to_buy_in_list = filtered_order[:to_buy_in]

            to_keep[to_buy_in_list] = True
            to_keep[to_sell_candidates] = False

            self.current_weight[to_sell_candidates] = 0.
            self.current_weight[to_keep] = 1.0 / self.target_num
            self._log(i, er, real_r, begin_positions, self.current_weight)

    def _log(self, step, er, real_r, begin_positions, end_positions):
        self.steps.append(step * np.ones(self.process.size, dtype=int))
        self.codes.append(np.array(list(range(1, self.process.size + 1))))
        self.ers.append(er)
        self.real_rs.append(real_r)
        self.previous_positions.append(begin_positions)
        self.current_positions.append(end_positions)

    def report(self):

        return pd.DataFrame({'date': np.concatenate(self.steps),
                             'code': np.concatenate(self.codes),
                             'er': np.concatenate(self.ers),
                             'dailyReturn': np.concatenate(self.real_rs),
                             'previous': np.concatenate(self.previous_positions),
                             'current': np.concatenate(self.current_positions)})


def analysis(report):
    df = report.copy(deep=True)

    df['realizedReturn'] = df.dailyReturn * df.current
    df['turn_over'] = np.abs(df.current - df.previous)
    equity_curve = df[['date', 'realizedReturn']].groupby('date').sum()
    turn_over = df[['date', 'turn_over']].groupby('date').sum()

    return equity_curve.cumsum(), turn_over.cumsum()


if __name__ == "__main__":
    sigma = 0.02
    num_assets = 500
    process = VectorAR1Process(0.05 * np.ones(num_assets),
                               0.9999,
                               (0.00001 ** 2) * np.diag(np.ones(num_assets)))
    process.evolve(2000)

    trading_plan = TradingPlan(process, 100, 101)
    trading_plan.simulate()

    report = trading_plan.report()
    equity_curve, turn_over = analysis(report)
    equity_curve.plot()
    turn_over.plot()

    trading_plan2 = TradingPlan(process, 100, 201)
    trading_plan2.simulate()
    report2 = trading_plan2.report()
    equity_curve2, turn_over2 = analysis(report2)
    equity_curve2.plot()
    turn_over2.plot()
    plt.show()

    #report2.to_excel('sample.xlsx')
