# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

from typing import List
from typing import Tuple
import numpy as np
import pandas as pd
from FactorModel.providers import Provider
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.portcalc import PortCalc
from FactorModel.schedule import Scheduler
from FactorModel.regulator import Constraints
from FactorModel.infokeeper import InfoKeeper
from FactorModel.regulator import Regulator
from FactorModel.facts import INDUSTRY_LIST
from FactorModel.facts import BENCHMARK
from FactorModel.settings import Settings


class Simulator(object):

    def __init__(self,
                 provider: Provider,
                 model_factory: ERModelTrainer,
                 cov_model: CovModel,
                 scheduler: Scheduler,
                 port_calc: PortCalc,
                 constraints_builder: Regulator) -> None:
        self.model_factory = model_factory
        self.provider = iter(provider)
        self.port_calc = port_calc
        self.info_keeper = InfoKeeper()
        self.scheduler = scheduler
        self.cov_model = cov_model
        self.constraints_builder = constraints_builder

    def simulate(self) -> pd.DataFrame:
        pre_holding = pd.DataFrame()

        while True:
            try:
                calc_date, apply_date, this_data = next(self.provider)
            except StopIteration:
                break

            print(apply_date)

            trading_constraints, _ = \
                self.constraints_builder.build_constraints(this_data)
            codes = this_data.code.astype(int)
            model = self.model_factory.fetch_model(apply_date)
            cov_matrix = self.cov_model.fetch_cov(calc_date, this_data)
            if not model.empty and len(cov_matrix) > 0:
                evolved_preholding, evolved_bm = \
                    Simulator.evolve_portfolio(codes, pre_holding, this_data)
                factor_values = \
                    this_data[self.model_factory.factor_names].as_matrix()
                er_model = model['model']
                er = er_model.calculate_er(factor_values)
                er_table = pd.DataFrame(er, index=codes, columns=['er'])
                adjusted_cov_matrix = \
                    self.parameters_adjust(er, cov_matrix,
                                           self.model_factory.decay)
                positions = self.rebalance(apply_date,
                                           er_table,
                                           evolved_preholding,
                                           cov=adjusted_cov_matrix,
                                           constraints=trading_constraints)

                self.aggregate_data(
                    er_table,
                    pre_holding,
                    evolved_preholding,
                    evolved_bm,
                    trading_constraints,
                    positions)
                self.log_info(apply_date, calc_date, positions)

                pre_holding = positions[['todayHolding']]

        return self.info_keeper.info_view()

    def parameters_adjust(self, er, cov, decay):
        cov_scaled = decay * cov
        return Settings.risk_aversion(er, cov_scaled) * cov_scaled

    def aggregate_data(self,
                       er_table: pd.DataFrame,
                       pre_holding: pd.DataFrame,
                       evolved_preholding: pd.DataFrame,
                       evolved_bm: pd.DataFrame,
                       trading_constraints: Constraints,
                       positions: pd.DataFrame) -> None:
        if not pre_holding.empty:
            positions['preHolding'] = pre_holding['todayHolding']
        else:
            positions['preHolding'] = 0.
        positions['er'] = er_table['er']
        positions['evolvedPreHolding'] = evolved_preholding['todayHolding']
        positions['evolvedBMWeight'] = evolved_bm[BENCHMARK]
        positions['suspend'] = trading_constraints.suspend

    def rebalance(self, apply_date, er_table, pre_holding, **kwargs):
        if self.scheduler.is_rebalance(apply_date):
            return self.port_calc.trade(er_table, pre_holding, **kwargs)
        else:
            return pre_holding.copy(deep=True)

    @staticmethod
    def evolve_portfolio(codes: List[int],
                         pre_holding: pd.DataFrame,
                         repo_data: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        values = repo_data[BENCHMARK].values.copy()
        values *= (1. + returns)
        values /= np.sum(values)
        evolved_bm = pd.DataFrame(
            data=values, index=codes, columns=[BENCHMARK])
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
