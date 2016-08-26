
# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import math
from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from FactorModel.optimizer import portfolio_optimizer
from FactorModel.regulator import Constraints


class PortCalc(object):

    def trade(self,
              er_table: pd.DataFrame,
              pre_holding: pd.DataFrame,
              **kwargs) -> pd.DataFrame:
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

        return Constraints(lb=lb, ub=ub, lc=constraints.lc, lct=constraints.lct, suspend=constraints.suspend)


class MeanVariancePortCalc(PortCalc):

    def __init__(self, method: str, cost_budget: Optional[float]=9999.) -> None:
        self.method = method
        self.cost_budget = cost_budget

    def trade(self,
              er_table: pd.DataFrame,
              pre_holding: pd.DataFrame,
              **kwargs) -> pd.DataFrame:
        rtntable = er_table.copy(deep=True)
        assets_number = len(er_table)
        er = er_table['er'].values
        cov = np.diag(0.0004 * np.ones(assets_number))

        constraints = kwargs['constraints']

        if np.sum(pre_holding['todayHolding']) > 0:
            cw = pre_holding['todayHolding']
            cost_budget = self.cost_budget
            constraints = PortCalc.adjust_constraints(pre_holding, constraints)
        else:
            cw = np.zeros(assets_number)
            cost_budget = 9999.
        tc = np.ones(assets_number) * 0.002

        weights, cost = portfolio_optimizer(cov=cov,
                                            er=er,
                                            tc=tc,
                                            cw=cw,
                                            constraints=constraints,
                                            method=self.method,
                                            cost_budget=cost_budget)
        rtntable['todayHolding'] = weights
        return rtntable


class EqualWeigthedPortCalc(PortCalc):

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

            sell_candiates = np.array((rank <= out_threshold) & (rtn_table['todayHolding'] > 0.))
            buy_candidates = np.array((rank >= in_threshold) & (rtn_table['todayHolding'] == 0.))

            total_sell_position = np.sum(rtn_table.loc[sell_candiates, 'todayHolding']) + max(1.0 - np.sum(rtn_table['todayHolding']), 0.)
            if total_sell_position == 0:
                return rtn_table

            rtn_table['rank'] = rank
            filter_table = rtn_table[buy_candidates].sort_values('rank', ascending=False)
            filter_table['todayHolding'] = 0.

            rtn_table.loc[sell_candiates, 'todayHolding'] = 0.
            to_buy_in = min(math.ceil(total_sell_position / 0.01), len(filter_table))
            to_buy_in_list = filter_table.index[:to_buy_in]
            filter_table.loc[to_buy_in_list, 'todayHolding'] = total_sell_position / to_buy_in

            rtn_table.loc[filter_table.index, 'todayHolding'] = filter_table['todayHolding'].values
            del rtn_table['rank']
        else:
            flags = np.array(rank >= in_threshold)
            active_assets = np.sum(flags)
            weights = np.zeros(total_assets)
            weights[flags] = 1.0 / active_assets
            rtn_table.loc[:, 'todayHolding'] = weights
        return rtn_table


class ERRankPortCalc(EqualWeigthedPortCalc):

    def __init__(self) -> None:
        pass

    def trade(self,
              er_table: pd.DataFrame,
              pre_holding: pd.DataFrame,
              **kwargs) -> pd.DataFrame:

        er_values = er_table['er'].values
        total_assets = len(er_table)
        rank = rankdata(er_values)
        in_threshold = total_assets - 100
        out_threshold = total_assets - 300

        return self.trade_by_cumstom_rank(er_table, pre_holding, in_threshold, out_threshold, rank)


class ERThresholdPortCalc(EqualWeigthedPortCalc):

    def __init__(self) -> None:
        pass

    def trade(self,
              er_table: pd.DataFrame,
              pre_holding: pd.DataFrame,
              **kwargs) -> pd.DataFrame:

        total_assets = len(er_table)
        in_threshold = np.percentile(er_table['er'], 100. * (total_assets - 100) / total_assets)
        out_threshold = 0.
        rank = er_table['er'].values

        return self.trade_by_cumstom_rank(er_table, pre_holding, in_threshold, out_threshold, rank)
