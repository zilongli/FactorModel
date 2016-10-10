# -*- coding: utf-8 -*-
u"""
Created on 2016-9-30

@author: cheng.li
"""

import math
import numpy as np
import pandas as pd


class VectorAR1Process(object):

    def __init__(self, initial, constants, phi, cov, seed=42):
        np.random.seed(seed)
        self.initial = np.array(initial)
        self.phi = phi
        self.constants = constants
        self.n = len(self.initial)

        self.start_point = self.initial
        self.end_point = None
        self.trajectory = []
        self.cholesky = np.linalg.cholesky(cov).T

    def evolve(self, nsteps):
        for i in range(nsteps):
            diffusion = np.random.randn(self.n) @ self.cholesky
            self.end_point = self.constants \
                + self.phi * self.start_point + diffusion
            self._push(self.start_point)
            self.start_point = self.end_point

        self._push(self.end_point)

    def _push(self, new_sample):
        self.trajectory.append(new_sample)

    @property
    def samples(self):
        return np.array(self.trajectory)

    @property
    def half_life(self):
        return - math.log(2) / math.log(self.phi)

    @property
    def size(self):
        return self.n

    def expectation(self, begin):
        return self.phi * begin


class TradingPlan(object):

    def __init__(self, ar1process, in_threshold, out_threshold):
        self.process = ar1process
        self.in_threshold = in_threshold
        self.out_threshold = out_threshold
        self.target_num = self.in_threshold
        self.current_weight = np.zeros(self.process.size)

        self.steps = []
        self.codes = []
        self.ers = []
        self.ranks = []
        self.real_rs = []
        self.previous_positions = []
        self.current_positions = []

    def simulate(self):
        trajectory = self.process.samples
        nsteps = np.size(trajectory, 0)
        for i in range(1, nsteps):
            begin_positions = self.current_weight.copy()
            start_point = trajectory[i-1, :]
            er = self.process.expectation(start_point)
            real_r = trajectory[i, :]

            to_sort = -er
            order = to_sort.argsort()
            ranks = order.argsort()

            to_buy_candidates = (ranks < self.in_threshold) \
                & (begin_positions == 0.)
            to_sell_candidates = (ranks >= self.out_threshold-1) \
                & (begin_positions != 0.)
            to_keep = (ranks < self.out_threshold-1) & (begin_positions != 0)

            to_buy_in = min(self.target_num - np.sum(to_keep),
                            np.sum(to_buy_candidates))

            filtered = to_sort.copy()
            filtered[~to_buy_candidates] = np.inf
            filtered_order = filtered.argsort()

            to_buy_in_list = filtered_order[:to_buy_in]

            to_keep[to_buy_in_list] = True
            to_keep[to_sell_candidates] = False

            end_positions = self.current_weight.copy()
            end_positions[to_sell_candidates] = 0.
            end_positions[to_keep] = 1.0 / self.target_num
            self.current_weight = end_positions
            self._log(i, er, ranks, real_r, begin_positions, self.current_weight)

    def _log(self, step, er, ranks, real_r, begin_positions, end_positions):
        self.steps.append(step * np.ones(self.process.size, dtype=int))
        self.codes.append(np.array(list(range(1, self.process.size + 1))))
        self.ers.append(er)
        self.ranks.append(ranks)
        self.real_rs.append(real_r)
        self.previous_positions.append(begin_positions)
        self.current_positions.append(end_positions)

    def report(self):

        return \
            pd.DataFrame({'date': np.concatenate(self.steps),
                          'code': np.concatenate(self.codes),
                          'er': np.concatenate(self.ers),
                          'ranks': np.concatenate(self.ranks),
                          'dailyReturn': np.concatenate(self.real_rs),
                          'previous': np.concatenate(self.previous_positions),
                          'current': np.concatenate(self.current_positions)})


def analysis(report):
    df = report.copy(deep=True)

    df['realizedReturn'] = df.dailyReturn * df.current
    df['turn_over'] = np.abs(df.current - df.previous) * 0.001
    equity_curve = df[['date', 'realizedReturn']].groupby('date').sum()
    turn_over = df[['date', 'turn_over']].groupby('date').sum()

    return equity_curve.sum()[0], turn_over.sum()[0]


def half_life_to_phi(half_life):
    return math.exp(-math.log(2) / half_life)


def turn_over_calculator(report):
    data_series = np.abs(report.current - report.previous) * 0.001
    data = np.sum(data_series)
    return data


def turn_over_curve(half_life, plans, calc, seed=42):
    phi = half_life_to_phi(half_life)
    var = 0.02 ** 2
    num_assets = 800
    process = VectorAR1Process(
        np.zeros(num_assets),
        np.zeros(num_assets),
        phi,
        (1. - phi ** 2) * var * np.diag(np.ones(num_assets)),
        seed)
    process.evolve(1000)

    data_list = []
    for plan in plans:
        trading_plan = TradingPlan(process, 100, plan)
        trading_plan.simulate()
        report = trading_plan.report()
        data = calc(report)
        data_list.append(data)

    series = pd.Series(data_list, plans)
    return series / series.iloc[0]

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    plans = list(range(100, 501, 50))
    series = turn_over_curve(0.5, plans)
    series.plot()
    plt.show()