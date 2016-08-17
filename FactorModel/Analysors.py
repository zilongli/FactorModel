# -*- coding: utf-8 -*-
u"""
Created on 2016-8-17

@author: cheng.li
"""

import pandas as pd


class PnLAnalyser(object):

    def __init__(self):
        self.report = None

    def calculate(self, env, extra_data):
        all_dates = extra_data.index.get_level_values(0).unique()
        everyday_return = []
        for date in all_dates:
            returns = env.fetch_values_from_repo(date, 'apply_date', ['nextReturn1day', 'zz500'])
            weights = extra_data.loc[date, 'weight']
            this_day_return = returns['nextReturn1day'].values.T @ (weights.values - returns['zz500'].values)
            everyday_return.append(this_day_return)
        return_table = pd.DataFrame(everyday_return, index=all_dates, columns=['pnl'])
        self.report = return_table
        return return_table.copy()

    def show(self):
        self.report.cumsum().plot()


if __name__ == "__main__":
    from FactorModel.Env import Env
    from FactorModel.utilities import load_mat
    from FactorModel.PortCalc import PortCalc
    from FactorModel.ERModel import ERModelTrainer
    from FactorModel.Simulator import Simulator

    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set_style('whitegrid')

    df = load_mat("d:/data.mat", rows=None)
    env = Env(df)
    trainer = ERModelTrainer(250, 1, 10)
    trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], df)
    port_calc = PortCalc()
    simulator = Simulator(env, trainer, port_calc)
    df = simulator.simulate()

    analysor = PnLAnalyser()
    returns = analysor.calculate(env, df)
    analysor.show()
    plt.show()