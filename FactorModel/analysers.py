# -*- coding: utf-8 -*-
u"""
Created on 2016-8-17

@author: cheng.li
"""

import numpy as np
import pandas as pd
from FactorModel.env import Env


class PnLAnalyser(object):

    def __init__(self):
        self.report = None

    def calculate(self,
                  env: Env,
                  extra_data: pd.DataFrame) -> pd.DataFrame:
        all_dates = extra_data.index.get_level_values(0).unique()
        everyday_return = []
        everyday_return_after_tc = []
        everyday_tc = []
        for date in all_dates:
            returns = env.fetch_values_from_repo(date, 'apply_date', ['nextReturn1day', 'zz500'])
            weights = extra_data.loc[date, 'todayHolding']
            pre_weights = extra_data.loc[date, 'preHolding']
            this_day_return = returns['nextReturn1day'].values.T @ (weights.values - returns['zz500'].values)
            this_day_tc = np.sum(np.abs(weights.values -pre_weights.values)) * 0.002
            everyday_return.append(this_day_return)
            everyday_tc.append(this_day_tc)
            everyday_return_after_tc.append(this_day_return - this_day_tc)
        return_table = pd.DataFrame(data=np.array([everyday_return, everyday_return_after_tc, everyday_tc]).T,
                                    index=all_dates,
                                    columns=['pnl', 'pnl - tc', 'tc'])
        self.report = return_table
        return return_table.copy()

    def plot(self) -> None:
        self.report.cumsum().plot()


if __name__ == "__main__":
    from FactorModel.utilities import load_mat
    from FactorModel.portcalc import MeanVariancePortCalc
    from FactorModel.ermodel import ERModelTrainer
    from FactorModel.simulator import Simulator

    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set_style('ticks')

    df = load_mat("d:/data.mat", rows=None)
    env = Env(df)
    trainer = ERModelTrainer(250, 1, 10)
    trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], df)
    port_calc = MeanVariancePortCalc('cost_budget', 2e-4)
    simulator = Simulator(env, trainer, port_calc)
    df = simulator.simulate()

    analysor = PnLAnalyser()
    returns = analysor.calculate(env, df)
    analysor.plot()
    plt.show()
