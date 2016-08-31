# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import numpy as np
import pandas as pd
from FactorModel.providers import Provider
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.portcalc import PortCalc
from FactorModel.infokeeper import InfoKeeper
from FactorModel.regulator import Regulator
from FactorModel.facts import INDUSTRY_LIST


class Simulator(object):

    def __init__(self,
                 provider: Provider,
                 model_factory: ERModelTrainer,
                 cov_model: CovModel,
                 port_calc: PortCalc) -> None:
        self.model_factory = model_factory
        self.provider = iter(provider)
        self.port_calc = port_calc
        self.info_keeper = InfoKeeper()
        self.cov_model = cov_model
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
            cov_matrix = self.cov_model.fetch_cov(calc_date, this_data)
            if not model.empty and len(cov_matrix) > 0:
                evolved_preholding = Simulator.evolve_portfolio(codes, pre_holding)
                factor_values = this_data[['Growth', 'CFinc1', 'Rev5m']].as_matrix()
                er = model['model'].calculate_er(factor_values)
                er_table = pd.DataFrame(er, index=codes, columns=['er'])
                positions = self.rebalance(apply_date,
                                           er_table,
                                           evolved_preholding,
                                           cov=cov_matrix, 
                                           constraints=trading_constraints)
                positions['preHolding'] = evolved_preholding['todayHolding']
                positions['suspend'] = trading_constraints.suspend
                self.log_info(apply_date, calc_date, positions)

                pre_holding = positions[['todayHolding']]

        return self.info_keeper.info_view()

    def rebalance(self, apply_date, er_table, pre_holding, **kwargs):
        return self.port_calc.trade(er_table, pre_holding, **kwargs)

    @staticmethod
    def evolve_portfolio(codes, pre_holding):
        evolved_preholding = pd.DataFrame(data=np.zeros(len(codes)), index=codes, columns=['todayHolding'])
        if not pre_holding.empty:
            evolved_preholding['todayHolding'] = pre_holding['todayHolding']
            evolved_preholding.fillna(0., inplace=True)
        return evolved_preholding

    def log_info(self,
                 apply_date: pd.Timestamp,
                 calc_date: pd.Timestamp,
                 data: pd.DataFrame) -> None:
        plain_data = data.reset_index()
        plain_data.index = [apply_date] * len(plain_data)
        plain_data.rename(columns={'level_1': 'code'}, inplace=True)
        plain_data['calcDate'] = [calc_date] * len(plain_data)
        self.info_keeper.attach_info(plain_data)
