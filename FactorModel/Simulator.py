# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

from typing import List
from typing import Any
import pandas as pd
from FactorModel.Env import Env
from FactorModel.ERModel import ERModelTrainer
from FactorModel.PortCalc import PortCalc


class Simulator(object):

    def __init__(self, env: Env, model_factory: ERModelTrainer, port_calc: PortCalc):
        self.model_factory = model_factory
        self.env = env
        self.port_calc = port_calc
        self.info_keeper = InfoKeeper()

    def simulate(self) -> pd.DataFrame:
        apply_dates = self.env.apply_dates()
        calc_dates = self.env.calc_dates()

        pre_holding = pd.DataFrame()

        for i, apply_date in enumerate(apply_dates):
            this_data = self.env.fetch_values_from_repo(apply_date)
            codes = this_data.code.astype(int)
            model = self.model_factory.fetch_model(apply_date)
            if not model.empty:
                factor_values = this_data[['Growth', 'CFinc1', 'Rev5m']].as_matrix()
                er = model.model.calculate_er(factor_values)
                er_table = pd.DataFrame(er, index=codes, columns=['er'])
                positions = self.port_calc.trade(er_table, pre_holding)
                if not pre_holding.empty:
                    positions['preHolding'] = pre_holding['todayHolding']
                    positions.fillna(0., inplace=True)
                else:
                    positions['preHolding'] = 0.0
                self.log_info(apply_date, calc_dates[i], positions)

                pre_holding = positions[['todayHolding']]

        return self.info_keeper.info_view()

    def log_info(self, apply_date: pd.Timestamp, calc_date: pd.Timestamp, data: pd.DataFrame) -> None:
        plain_data = data.reset_index()
        plain_data.index = [apply_date] * len(plain_data)
        plain_data.rename(columns={'level_1': 'code'}, inplace=True)
        plain_data['calcDate'] = [calc_date] * len(plain_data)
        self.info_keeper.attach_info(plain_data)


class InfoKeeper(object):

    def __init__(self):
        self.data_sets = []
        self.stored_data = pd.DataFrame()
        self.current_index = 0

    def attach_info(self, appended_data: pd.DataFrame) -> None:
        self.data_sets.append(appended_data)

    def info_view(self) -> pd.DataFrame:
        if self.current_index < len(self.data_sets):
            self.stored_data = self.stored_data.append(self.data_sets[self.current_index:])
            self.current_index = len(self.data_sets)
        return self.stored_data.copy(deep=True)


if __name__ == "__main__":

    from FactorModel.utilities import load_mat
    from FactorModel.PortCalc import ThresholdPortCalc, RankPortCalc
    df = load_mat("d:/data.mat", rows=220000)
    env = Env(df)
    trainer = ERModelTrainer(250, 1, 10)
    trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], df)
    port_calc = RankPortCalc()
    simulator = Simulator(env, trainer, port_calc)
    df = simulator.simulate()
    print(df)
