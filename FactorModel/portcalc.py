
# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import abc
import math
from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from FactorModel.optimizer import CostBudgetProblem
from FactorModel.optimizer import NoCostProblem
from FactorModel.regulator import Constraints
from FactorModel.covmodel import CovModel
from FactorModel.ermodel import ERModelTrainer
from FactorModel.regulator import Regulator
from FactorModel.schedule import Scheduler


class PortCalc(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def trade(self,
              calc_date: pd.Timestamp,
              apply_data: pd.Timestamp,
              pre_holding: pd.DataFrame,
              repo_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @staticmethod
    def adjust_constraints(pre_holding: pd.DataFrame,
                           constraints: Constraints) -> Constraints:
        lb = constraints.lb
        ub = constraints.ub
        weights = pre_holding['todayHolding'].values
        weights[np.abs(weights) < 1e-8] = 0.

        lb[constraints.suspend] = weights[constraints.suspend]
        ub[constraints.suspend] = weights[constraints.suspend]

        return Constraints(
            lb=lb,
            ub=ub,
            lc=constraints.lc,
            suspend=constraints.suspend)


class MeanVariancePortCalc(PortCalc):

    def __init__(self,
                 method: str,
                 model_factory: ERModelTrainer,
                 cov_model: CovModel,
                 constraints_builder: Regulator,
                 scheduler: Scheduler,
                 cost_budget: Optional[float]=9999.) -> None:
        self.method = method
        self.model_factory = model_factory
        self.cov_model = cov_model
        self.constraints_builder = constraints_builder
        self.scheduler = scheduler
        self.cost_budget = cost_budget

    def trade(self,
              calc_date: pd.Timestamp,
              apply_date: pd.Timestamp,
              pre_holding: pd.DataFrame,
              repo_data: pd.DataFrame) -> pd.DataFrame:

        model = self.model_factory.fetch_model(apply_date)
        cov_matrix = self.cov_model.fetch_cov(calc_date, repo_data)

        if not model.empty and len(cov_matrix) > 0:
            factor_values = \
                repo_data[self.model_factory.factor_names].as_matrix()
            er_model = model['model']
            er = er_model.calculate_er(factor_values)
            codes = repo_data.code.astype(int)
            er_table = pd.DataFrame(er, index=codes, columns=['er'])
            if self.scheduler.is_rebalance(apply_date):
                adjusted_cov_matrix = cov_matrix * self.model_factory.decay

                rtntable = er_table.copy(deep=True)
                assets_number = len(er_table)

                constraints, _ = \
                    self.constraints_builder.build_constraints(repo_data)

                if np.sum(pre_holding['todayHolding']) > 0:
                    cw = pre_holding['todayHolding']
                    cost_budget = self.cost_budget
                else:
                    cw = np.zeros(assets_number)
                    cost_budget = 9999.
                constraints = \
                    PortCalc.adjust_constraints(pre_holding, constraints)

                tc = np.ones(assets_number) * 0.002

                if self.method == 'no_cost':
                    prob = NoCostProblem(adjusted_cov_matrix,
                                         er,
                                         constraints)
                else:
                    prob = CostBudgetProblem(adjusted_cov_matrix,
                                             er,
                                             constraints,
                                             tc,
                                             cost_budget)
                weights, cost = prob.optimize(cw)

                rtntable['todayHolding'] = weights
                return er_table, rtntable
            else:
                return er_table, pre_holding.copy(deep=True)
        else:
            return pd.DataFrame(), pre_holding.copy(deep=True)


class EqualWeigthedPortCalc(PortCalc):

    def __init__(self,
                 model_factory: ERModelTrainer,
                 scheduler: Scheduler,
                 ) -> None:
        self.model_factory = model_factory
        self.scheduler = scheduler

    def trade_by_cumstom_rank(self,
                              er_table: pd.DataFrame,
                              pre_holding: pd.DataFrame,
                              in_threshold: float,
                              out_threshold: float,
                              rank: np.array) -> pd.DataFrame:

        rtn_table = er_table.copy(deep=True)
        total_assets = len(rtn_table)

        if np.sum(pre_holding['todayHolding']) > 0:
            # match pre_holding to today's codes
            rtn_table.loc[:, 'todayHolding'] = 0.
            rtn_table['todayHolding'] = pre_holding['todayHolding']

            sell_candiates = np.array(
                (rank <= out_threshold) & (rtn_table['todayHolding'] > 0.))
            buy_candidates = np.array(
                (rank >= in_threshold) & (rtn_table['todayHolding'] == 0.))

            rtn_table['rank'] = rank
            filter_table = rtn_table[buy_candidates] \
                .sort_values('rank', ascending=False)
            filter_table['todayHolding'] = 0.

            rtn_table.loc[sell_candiates, 'todayHolding'] = 0.

            if not self.rebalance:
                total_sell_position = \
                    np.sum(rtn_table.loc[sell_candiates, 'todayHolding']) \
                    + max(1.0 - np.sum(rtn_table['todayHolding']), 0.)
                if total_sell_position == 0:
                    return rtn_table

                to_buy_in = min(
                    math.ceil(total_sell_position / 0.01),
                    len(filter_table))
                to_buy_in_list = filter_table.index[:to_buy_in]

                filter_table.loc[to_buy_in_list, 'todayHolding'] = \
                    total_sell_position / to_buy_in
                rtn_table.loc[filter_table.index, 'todayHolding'] = \
                    filter_table['todayHolding'].values
            else:
                to_keep = np.array(
                    (rank > out_threshold) & (rtn_table['todayHolding'] > 0.))
                to_buy_in = min(100 - np.sum(to_keep), len(filter_table))
                to_buy_in_list = list(filter_table.index[:to_buy_in])
                to_keep_list = list(rtn_table.index[to_keep])
                basket = to_buy_in_list + to_keep_list
                rtn_table.loc[basket, 'todayHolding'] = 1.0 / len(basket)
            del rtn_table['rank']
        else:
            flags = np.array(rank >= in_threshold)
            active_assets = np.sum(flags)
            weights = np.zeros(total_assets)
            weights[flags] = 1.0 / active_assets
            rtn_table.loc[:, 'todayHolding'] = weights
        return rtn_table


class ERRankPortCalc(EqualWeigthedPortCalc):

    def __init__(self,
                 in_threshold: int,
                 out_threshold: int,
                 model_factory: ERModelTrainer,
                 scheduler: Scheduler,
                 rebalance: Optional[bool]=True) -> None:
        super().__init__(model_factory, scheduler)
        self.in_threshold = in_threshold
        self.out_threshold = out_threshold
        self.rebalance = rebalance

    def trade(self,
              calc_date: pd.Timestamp,
              apply_date: pd.Timestamp,
              pre_holding: pd.DataFrame,
              repo_data: pd.DataFrame) -> pd.DataFrame:
        model = self.model_factory.fetch_model(apply_date)

        if not model.empty:
            codes = repo_data.code.astype(int)
            er_model = model['model']
            factor_values = \
                repo_data[self.model_factory.factor_names].as_matrix()
            er = er_model.calculate_er(factor_values)
            er_table = pd.DataFrame(er, index=codes, columns=['er'])
            if self.scheduler.is_rebalance(apply_date):
                er_values = er_table['er'].values
                total_assets = len(er_table)
                rank = rankdata(er_values)
                in_threshold = total_assets - self.in_threshold + 1
                out_threshold = total_assets - self.out_threshold + 1

                positions = self.trade_by_cumstom_rank(
                    er_table, pre_holding, in_threshold, out_threshold, rank)
                return er_table, positions
            else:
                return er_table, pre_holding.copy(deep=True)
        else:
            return pd.DataFrame(), pre_holding.copy(deep=True)


class ERThresholdPortCalc(EqualWeigthedPortCalc):

    def __init__(self,
                 in_threshold: float,
                 out_threshold: float,
                 model_factory: ERModelTrainer,
                 scheduler: Scheduler,
                 rebalance: Optional[bool]=True) -> None:
        super().__init__(model_factory, scheduler)
        self.in_threshold = in_threshold
        self.out_threshold = out_threshold
        self.rebalance = rebalance

    def trade(self,
              calc_date: pd.Timestamp,
              apply_date: pd.Timestamp,
              pre_holding: pd.DataFrame,
              repo_data: pd.DataFrame) -> pd.DataFrame:
        model = self.model_factory.fetch_model(apply_date)
        if not model.empty:
            codes = repo_data.code.astype(int)
            er_model = model['model']
            factor_values = \
                repo_data[self.model_factory.factor_names].as_matrix()
            er = er_model.calculate_er(factor_values)
            er_table = pd.DataFrame(er, index=codes, columns=['er'])
            if self.scheduler.is_rebalance(apply_date):
                in_threshold = self.in_threshold
                out_threshold = self.out_threshold
                rank = er_table['er'].values

                positions = self.trade_by_cumstom_rank(
                    er_table, pre_holding, in_threshold, out_threshold, rank)
                return er_table, positions
            else:
                return er_table, pre_holding.copy(deep=True)
        else:
            return pd.DataFrame(), pre_holding.copy(deep=True)
