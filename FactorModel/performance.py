u"""
Created on 2016-9-5

@author: cheng.li
"""

import abc
import pandas as pd
import numpy as np
from FactorModel.schedule import Scheduler
from FactorModel.portcalc import PortCalc
from FactorModel.ermodel import ERModelTrainer
from FactorModel.regulator import Regulator
from FactorModel.facts import BENCHMARK


class PerfAttributeBase(metaclass=abc.ABCMeta):

    def __init__(self):
        self.p_table = pd.DataFrame()
        self.report = pd.DataFrame()

    @abc.abstractmethod
    def _evolve(self,
                codes,
                today_holding,
                evolved_bm,
                evolved_new_table):
        pass

    @abc.abstractmethod
    def _rebalance(self,
                   codes,
                   today_holding,
                   pre_holding,
                   evolved_bm,
                   factor_values,
                   factor_names,
                   er_model,
                   port_calc):
        pass

    def save_data(self,
                  codes,
                  calc_date,
                  apply_date,
                  today_holding,
                  p_table,
                  p_matrix,
                  evolved_bm,
                  total_pnl,
                  factor_pnl,
                  factor_names):
        if self.report.empty:
            num_cols = [s + '_num' for s in ['total'] + list(factor_names)]
            weights_cols = [
                s + '_weight' for s
                in [BENCHMARK, 'total'] + list(factor_names)]
            col_names = \
                ['calcDate', 'total'] \
                + list(factor_names) \
                + num_cols + weights_cols
            self.report = pd.DataFrame(columns=col_names)
        num_total = np.sum(today_holding != 0)
        num_factors = np.sum(p_matrix != 0, axis=0)
        bm_weight = np.sum(evolved_bm)
        total_weight = np.sum(today_holding)
        factor_weight = np.sum(p_matrix, axis=0)
        self.report.loc[apply_date] = \
            [calc_date, total_pnl] \
            + list(factor_pnl) \
            + [num_total] \
            + list(num_factors) \
            + [bm_weight, total_weight] + list(factor_weight)

    def analysis(self,
                 er_trainer: ERModelTrainer,
                 schedule: Scheduler,
                 port_calc: PortCalc,
                 cov_model,
                 constraints_builder: Regulator,
                 data: pd.DataFrame) -> None:
        all_apply_dates = sorted(data.index.unique())
        all_calculate_dates = sorted(data.calcDate.unique())
        factor_names = er_trainer.factor_names
        col_names = ['calcDate', 'total'] + list(factor_names)
        self.report = pd.DataFrame(columns=col_names)
        for calc_date, apply_date in zip(all_calculate_dates, all_apply_dates):
            print(calc_date, apply_date)

            if self.p_table.empty and not schedule.is_rebalance(apply_date):
                continue

            this_data = data.loc[apply_date, :]
            codes = this_data.code
            trading_constraints, _ = \
                constraints_builder.build_constraints(this_data)
            cov_matrix = cov_model.fetch_cov(calc_date, this_data)
            returns = this_data['nextReturn1day'].values
            today_holding = this_data['todayHolding'].values
            evolved_bm = this_data['evolvedBMWeight'].values

            if not schedule.is_rebalance(apply_date):
                evolved_new_table = pd.DataFrame(
                    np.zeros((len(codes), len(factor_names)), dtype=float),
                    index=codes,
                    columns=factor_names)
                evolved_new_table[factor_names] = self.p_table[factor_names]
                evolved_new_table.fillna(0, inplace=True)
                cashes = 1. - evolved_new_table.sum()
                evolved_new_table = \
                    evolved_new_table.multiply(1. + returns, axis=0)
                evolved_new_table /= cashes + evolved_new_table.sum()
                total_pnl = np.dot(today_holding, returns) \
                    - np.sum(today_holding) / np.sum(evolved_bm) \
                    * np.dot(evolved_bm, returns)
                p_matrix = self._evolve(codes,
                                        today_holding,
                                        evolved_bm,
                                        evolved_new_table)
                factor_pnl = returns @ p_matrix \
                    - np.sum(p_matrix, axis=0) \
                    / np.sum(evolved_bm) * np.dot(evolved_bm, returns)
                self.p_table = evolved_new_table
            else:
                evolved_preholding = this_data['evolvedPreHolding'].values
                pre_holding = pd.DataFrame(
                    evolved_preholding, index=codes, columns=['todayHolding'])
                factor_values = this_data[factor_names]
                er_model = er_trainer.fetch_model(apply_date)['model']

                total_pnl = np.dot(today_holding, returns) \
                    - np.sum(today_holding) \
                    / np.sum(evolved_bm) * np.dot(evolved_bm, returns)
                p_holding, p_matrix = self._rebalance(codes,
                                                      today_holding,
                                                      pre_holding,
                                                      evolved_bm,
                                                      factor_values,
                                                      factor_names,
                                                      er_model,
                                                      port_calc,
                                                      cov_matrix,
                                                      trading_constraints)
                factor_pnl = returns @ p_matrix \
                    - np.sum(p_matrix, axis=0) \
                    / np.sum(evolved_bm) * np.dot(evolved_bm, returns)
                self.p_table = pd.DataFrame(
                    p_holding, index=codes, columns=factor_names)
            self.save_data(codes,
                           calc_date,
                           apply_date,
                           today_holding,
                           self.p_table,
                           p_matrix,
                           evolved_bm,
                           total_pnl,
                           factor_pnl,
                           factor_names)

    def plot(self):
        self.report[self.report.columns[1:5]].cumsum().plot()


class PerfAttributeLOO(PerfAttributeBase):

    def __init__(self):
        super().__init__()

    def _evolve(self,
                codes,
                today_holding,
                evolved_bm,
                evolved_new_table):
        evolved_factor_p = evolved_new_table.values
        tmp = today_holding.copy()
        tmp.shape = -1, 1
        return tmp - evolved_factor_p

    def _rebalance(self,
                   codes,
                   today_holding,
                   pre_holding,
                   evolved_bm,
                   factor_values,
                   factor_names,
                   er_model,
                   port_calc,
                   cov_matrix,
                   trading_constraints):
        p_holding = np.zeros((len(codes), len(factor_names)), dtype=float)
        for i, factor in enumerate(factor_names):
            tb_copy = factor_values.copy(deep=True)
            tb_copy[factor] = 0.
            er = er_model.calculate_er(tb_copy)
            er_table = pd.DataFrame(er, index=codes, columns=['er'])
            res = port_calc.trade(er_table,
                                  pre_holding,
                                  cov=cov_matrix,
                                  constraints=trading_constraints)
            p_holding[:, i] = res['todayHolding'].values
        tmp = today_holding.copy()
        tmp.shape = -1, 1
        return p_holding, tmp - p_holding


class PerfAttributeAOI(PerfAttributeBase):

    def __init__(self):
        super().__init__()

    def _evolve(self,
                codes,
                today_holding,
                evolved_bm,
                evolved_new_table):
        evolved_factor_p = evolved_new_table.values
        return evolved_factor_p

    def _rebalance(self,
                   codes,
                   today_holding,
                   pre_holding,
                   evolved_bm,
                   factor_values,
                   factor_names,
                   er_model,
                   port_calc,
                   cov_matrix,
                   trading_constraints):
        p_holding = np.zeros((len(codes), len(factor_names)), dtype=float)
        for i, factor in enumerate(factor_names):
            tb_copy = factor_values.copy(deep=True)
            tb_copy.loc[:, :] = 0.
            tb_copy[factor] = factor_values[factor]
            er = er_model.calculate_er(tb_copy)
            er_table = pd.DataFrame(er, index=codes, columns=['er'])
            res = port_calc.trade(er_table,
                                  pre_holding,
                                  cov=cov_matrix,
                                  constraints=trading_constraints)
            p_holding[:, i] = res['todayHolding'].values
        return p_holding, p_holding


class PerfAttributeFocusLOO(PerfAttributeBase):

    def __init__(self):
        super().__init__()

    def _evolve(self,
                codes,
                today_holding,
                evolved_bm,
                evolved_new_table):
        null_assets = np.array(today_holding == 0)
        evolved_factor_p = evolved_new_table.values.copy()
        evolved_factor_p[null_assets, :] = 0.
        tmp = today_holding.copy()
        tmp.shape = -1, 1
        return tmp - evolved_factor_p

    def _rebalance(self,
                   codes,
                   today_holding,
                   pre_holding,
                   evolved_bm,
                   factor_values,
                   factor_names,
                   er_model,
                   port_calc,
                   cov_matrix,
                   trading_constraints):
        null_assets = np.array(today_holding == 0)
        p_holding = np.zeros((len(codes), len(factor_names)), dtype=float)
        for i, factor in enumerate(factor_names):
            tb_copy = factor_values.copy(deep=True)
            tb_copy[factor] = 0.
            er = er_model.calculate_er(tb_copy)
            er_table = pd.DataFrame(er, index=codes, columns=['er'])
            res = port_calc.trade(er_table,
                                  pre_holding,
                                  cov=cov_matrix,
                                  constraints=trading_constraints)
            p_holding[:, i] = res['todayHolding'].values
        filtered_p_holding = p_holding.copy()
        filtered_p_holding[null_assets, :] = 0.
        tmp = today_holding.copy()
        tmp.shape = -1, 1
        return p_holding, tmp - filtered_p_holding


class PerfAttributeFocusAOI(PerfAttributeBase):

    def __init__(self):
        super().__init__()

    def _evolve(self,
                codes,
                today_holding,
                evolved_bm,
                evolved_new_table):
        null_assets = np.array(today_holding == 0)
        evolved_factor_p = evolved_new_table.values.copy()
        evolved_factor_p[null_assets, :] = 0.
        return evolved_factor_p

    def _rebalance(self,
                   codes,
                   today_holding,
                   pre_holding,
                   evolved_bm,
                   factor_values,
                   factor_names,
                   er_model,
                   port_calc,
                   cov_matrix,
                   trading_constraints):
        null_assets = np.array(today_holding == 0)
        p_holding = np.zeros((len(codes), len(factor_names)), dtype=float)
        for i, factor in enumerate(factor_names):
            tb_copy = factor_values.copy(deep=True)
            tb_copy.loc[:, :] = 0.
            tb_copy[factor] = factor_values[factor]
            er = er_model.calculate_er(tb_copy)
            er_table = pd.DataFrame(er, index=codes, columns=['er'])
            res = port_calc.trade(er_table,
                                  pre_holding,
                                  cov=cov_matrix,
                                  constraints=trading_constraints)
            p_holding[:, i] = res['todayHolding'].values
        filtered_p_holding = p_holding.copy()
        filtered_p_holding[null_assets, :] = 0.
        return p_holding, filtered_p_holding
