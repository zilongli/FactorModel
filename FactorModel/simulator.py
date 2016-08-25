# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import numpy as np
import pandas as pd
from FactorModel.providers import Provider
from FactorModel.ermodel import ERModelTrainer
from FactorModel.portcalc import PortCalc
from FactorModel.infokeeper import InfoKeeper
from FactorModel.regulator import Regulator
from FactorModel.facts import INDUSTRY_LIST


class Simulator(object):

    def __init__(self,
                 provider: Provider,
                 model_factory: ERModelTrainer,
                 port_calc: PortCalc) -> None:
        self.model_factory = model_factory
        self.provider = iter(provider)
        self.port_calc = port_calc
        self.info_keeper = InfoKeeper()
        self.constraints_builder = Regulator(INDUSTRY_LIST)

    def simulate(self) -> pd.DataFrame:
        pre_holding = pd.DataFrame()

        while True:
            try:
                calc_date, apply_date, this_data = next(self.provider)
            except StopIteration:
                break

            print(apply_date)

            trading_constraints, _ = self.constraints_builder.build_constraints(this_data)
            codes = this_data.code.astype(int)
            model = self.model_factory.fetch_model(apply_date)
            if not model.empty:
                matched_preholding = pd.DataFrame(data=np.zeros(len(codes)), index=codes, columns=['todayHolding'])
                if not pre_holding.empty:
                    matched_preholding['todayHolding'] = pre_holding['todayHolding']
                    matched_preholding.fillna(0., inplace=True)
                factor_values = this_data[['Growth', 'CFinc1', 'Rev5m']].as_matrix()
                er = model['model'].calculate_er(factor_values)
                er_table = pd.DataFrame(er, index=codes, columns=['er'])
                positions = self.port_calc.trade(er_table, matched_preholding, constraints=trading_constraints)
                positions['preHolding'] = matched_preholding['todayHolding']
                positions['suspend'] = trading_constraints.suspend
                self.log_info(apply_date, calc_date, positions)

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
    from FactorModel.providers import FileProvider
    from FactorModel.portcalc import MeanVariancePortCalc

    env = FileProvider("/home/wegamekinglc/Downloads/data.mat", rows=220000)
    trainer = ERModelTrainer(250, 1, 10)
    trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], env.source_data)
    portcalc = MeanVariancePortCalc('cost_budget', 2e-4)
    simulator = Simulator(env, trainer, portcalc)
    df = simulator.simulate()
    print(df)
