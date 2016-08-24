# -*- coding: utf-8 -*-
u"""
Created on 2016-8-17

@author: cheng.li
"""

import numpy as np
import pandas as pd


class PnLAnalyser(object):

    def __init__(self) -> None:
        self.report = None

    def calculate(self,
                  data: pd.DataFrame) -> pd.DataFrame:
        all_dates = data.index.get_level_values(0).unique()
        everyday_return = []
        everyday_return_after_tc = []
        everyday_tc = []
        for date in all_dates:
            returns = data.loc[date, ['nextReturn1day', 'zz500']]
            weights = data.loc[date, 'todayHolding']
            pre_weights = data.loc[date, 'preHolding']
            this_day_return = returns['nextReturn1day'].values.T @ (weights.values - returns['zz500'].values)
            this_day_tc = np.sum(np.abs(weights.values - pre_weights.values)) * 0.002
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
    from FactorModel.portcalc import MeanVariancePortCalc
    from FactorModel.ermodel import ERModelTrainer
    from FactorModel.simulator import Simulator
    from FactorModel.providers import FileProvider

    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set_style('ticks')

    env = FileProvider("/home/wegamekinglc/Downloads/data.mat", rows=None)
    trainer = ERModelTrainer(250, 1, 10)
    trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], env.source_data)
    port_calc = MeanVariancePortCalc('cost_budget', 2e-4)
    simulator = Simulator(env, trainer, port_calc)
    df = simulator.simulate()

    analyser = PnLAnalyser()

    raw_data = env.source_data
    raw_data = raw_data.set_index('code', append=True)
    df = df.set_index('code', append=True)
    raw_data[df.columns] = df
    raw_data.dropna(inplace=True)
    returns = analyser.calculate(raw_data)
    analyser.plot()
    plt.show()
