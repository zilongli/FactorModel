# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

from typing import List
from typing import Any
import numpy as np
import pandas as pd
from FactorModel.env import Env
from FactorModel.ermodel import ERModelTrainer
from FactorModel.portcalc import PortCalc
from FactorModel.infokeeper import InfoKeeper
from FactorModel.regulator import Regulator
from FactorModel.facts import INDUSTRY_LIST


class Simulator(object):

    def __init__(self,
                 env: Env,
                 model_factory: ERModelTrainer,
                 port_calc: PortCalc) -> None:
        self.model_factory = model_factory
        self.env = env
        self.port_calc = port_calc
        self.info_keeper = InfoKeeper()
        self.constraints_builder = Regulator(INDUSTRY_LIST)

    def simulate(self) -> pd.DataFrame:
        apply_dates = self.env.apply_dates()
        calc_dates = self.env.calc_dates()

        pre_holding = pd.DataFrame()

        for i, apply_date in enumerate(apply_dates):
            print(apply_date)
            this_data = self.env.fetch_values_from_repo(apply_date)
            trading_constraints, _ = self.constraints_builder.build_constraints(this_data)
            codes = this_data.code.astype(int)
            model = self.model_factory.fetch_model(apply_date)
            if not model.empty:
                factor_values = this_data[['Growth', 'CFinc1', 'Rev5m']].as_matrix()
                er = model['model'].calculate_er(factor_values)
                er_table = pd.DataFrame(er, index=codes, columns=['er'])
                positions = self.port_calc.trade(er_table, pre_holding, constraints=trading_constraints)
                if not pre_holding.empty:
                    positions['preHolding'] = pre_holding['todayHolding']
                    positions.fillna(0., inplace=True)
                else:
                    positions['preHolding'] = 0.0
                self.log_info(apply_date, calc_dates[i], positions)

                pre_holding = positions[['todayHolding']]

        return self.info_keeper.info_view()

    def log_info(self,
                 apply_date: pd.Timestamp,
                 calc_date: pd.Timestamp,
                 data: pd.DataFrame) -> None:
        plain_data = data.reset_index()
        plain_data.index = [apply_date] * len(plain_data)
        plain_data.rename(columns={'level_1': 'code'}, inplace=True)
        plain_data['calcDate'] = [calc_date] * len(plain_data)
        self.info_keeper.attach_info(plain_data)


if __name__ == "__main__":
    from FactorModel.utilities import load_mat
    from FactorModel.portcalc import MeanVariancePortCalc
    df = load_mat("/home/wegamekinglc/Downloads/data.mat", rows=220000)
    env = Env(df)
    trainer = ERModelTrainer(250, 1, 10)
    trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], df)
    print(trainer.models)
    port_calc = MeanVariancePortCalc('cost_budget', 2e-4)
    simulator = Simulator(env, trainer, port_calc)
    df = simulator.simulate()
    print(df)
