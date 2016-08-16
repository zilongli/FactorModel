# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import pandas as pd

class Simulator(object):

    def __init__(self, env, model_factory, port_calc):
        self.model_factory = model_factory
        self.env = env
        self.port_calc = port_calc

    def simulate(self):
        apply_dates = self.env.apply_dates()
        calc_dates = self.env.calc_dates()

        for i, apply_date in enumerate(apply_dates):
            thisData = self.env.fetch_values_from_repo(apply_date)
            codes = thisData.code.astype(int)
            model = self.model_factory.fetch_model(apply_date)
            if not model.empty:
                factor_values = thisData[['Growth', 'CFinc1', 'Rev5m']].as_matrix()
                er = model.model.calculateER(factor_values)
                er_table = pd.DataFrame(er, index=codes, columns=['er'])
                positions = self.port_calc.trade(er_table)


if __name__ == "__main__":
    from utilities import load_mat
    from Env import Env
    from ERModel import ERModelTrainer
    from PortCalc import PortCalc
    df = load_mat("d:/data.mat")#[:200000]
    env = Env(df)
    trainer = ERModelTrainer(250, 1, 10)
    trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], df)
    port_calc = PortCalc()
    simulator = Simulator(env, trainer, port_calc)
    simulator.simulate()
