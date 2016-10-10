# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

from typing import Tuple
import numpy as np
import pandas as pd
from FactorModel.providers import Provider
from FactorModel.portcalc import PortCalc
from FactorModel.infokeeper import InfoKeeper


class Simulator(object):

    def __init__(self,
                 provider: Provider,
                 port_calc: PortCalc) -> None:
        self.provider = iter(provider)
        self.port_calc = port_calc
        self.info_keeper = InfoKeeper()

    def simulate(self) -> pd.DataFrame:
        pre_holding = pd.DataFrame()

        while True:
            try:
                calc_date, apply_date, this_data = next(self.provider)
            except StopIteration:
                break

            print(apply_date)

            evolved_preholding, evolved_bm = \
                Simulator.evolve_portfolio(pre_holding, this_data)

            er_table, positions = self.rebalance(calc_date,
                                                 apply_date,
                                                 evolved_preholding,
                                                 this_data)

            if not er_table.empty and not pre_holding.empty:
                self.aggregate_data(
                        er_table,
                        pre_holding,
                        evolved_preholding,
                        evolved_bm,
                        positions)
                self.log_info(apply_date, calc_date, positions)

            pre_holding = positions[['todayHolding']]

        return self.info_keeper.info_view()

    def aggregate_data(self,
                       er_table: pd.DataFrame,
                       pre_holding: pd.DataFrame,
                       evolved_preholding: pd.DataFrame,
                       evolved_bm: pd.DataFrame,
                       positions: pd.DataFrame) -> None:
        if not pre_holding.empty:
            positions['preHolding'] = pre_holding['todayHolding']
        else:
            positions['preHolding'] = 0.
        positions['er'] = er_table['er']
        positions['evolvedPreHolding'] = evolved_preholding['todayHolding']
        positions['evolvedBMWeight'] = evolved_bm['benchmark']

    def rebalance(self,
                  calc_date: pd.Timestamp,
                  apply_date: pd.Timestamp,
                  pre_holding: pd.DataFrame,
                  this_data: pd.DataFrame):
        return self.port_calc.trade(calc_date,
                                    apply_date,
                                    pre_holding,
                                    this_data)

    @staticmethod
    def evolve_portfolio(pre_holding: pd.DataFrame,
                         repo_data: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        codes = repo_data.code.astype(int)
        evolved_preholding = pd.DataFrame(
            data=np.zeros(len(codes)), index=codes, columns=['todayHolding'])
        returns = repo_data['dailyReturn'].values
        if not pre_holding.empty:
            evolved_preholding['todayHolding'] = pre_holding['todayHolding']
            evolved_preholding.fillna(0., inplace=True)
            values = evolved_preholding['todayHolding'].values
            cash = 1. - np.sum(evolved_preholding['todayHolding'])
            values *= (1. + returns)
            values /= cash + np.sum(values)

        values = repo_data['benchmark'].values.copy()
        values *= (1. + returns)
        values /= np.sum(values)
        evolved_bm = pd.DataFrame(
            data=values, index=codes, columns=['benchmark'])
        return evolved_preholding, evolved_bm

    def log_info(self,
                 apply_date: pd.Timestamp,
                 calc_date: pd.Timestamp,
                 data: pd.DataFrame) -> None:
        plain_data = data.reset_index()
        plain_data.index = [apply_date] * len(plain_data)
        plain_data.rename(columns={'level_1': 'code'}, inplace=True)
        plain_data['calcDate'] = [calc_date] * len(plain_data)
        self.info_keeper.attach_info(plain_data)
