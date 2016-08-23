# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import math
import pandas as pd
import numpy as np


class PortCalc(object):
    pass


class MeanVariancePortCalc(PortCalc):

    def __init__(self):
        pass

    def trade(self, er_table: pd.DataFrame, pre_holding: pd.DataFrame) -> pd.DataFrame:
        pass


class RankPortCalc(PortCalc):

    def __init__(self):
        pass

    def trade(self, er_table: pd.DataFrame, pre_holding: pd.DataFrame) -> pd.DataFrame:
        rtntable = er_table.copy(deep=True)

        total_assets = len(rtntable)
        level = 100
        rank_threshold = 200

        if not pre_holding.empty:
            # match pre_holding to today's codes
            rtntable.loc[:, 'todayHolding'] = 0.
            rtntable['todayHolding'] = pre_holding['todayHolding']

            rtntable.sort_values('er', ascending=False, inplace=True)
            rtntable.loc[:, 'rank'] = range(1, len(rtntable) + 1)
            rank_diff = rtntable['rank'].values - level

            sell_candiates = np.array((rank_diff >= rank_threshold) & (rtntable['todayHolding'] > 0.))
            buy_candidates = np.array((rank_diff <= 0) & (rtntable['todayHolding'] == 0.))

            total_sell_position = np.sum(rtntable.loc[sell_candiates, 'todayHolding']) + max(1.0 - np.sum(rtntable['todayHolding']), 0.)
            if total_sell_position == 0:
                return rtntable

            filter_table = rtntable[buy_candidates].copy(deep=True)
            filter_table['todayHolding'] = 0.

            rtntable.loc[sell_candiates, 'todayHolding'] = 0.
            to_buy_in = min(math.ceil(total_sell_position / 0.01), len(filter_table))
            to_buy_in_list = filter_table.index[:to_buy_in]
            filter_table.loc[to_buy_in_list, 'todayHolding'] = total_sell_position / to_buy_in
            rtntable.loc[filter_table.index, 'todayHolding'] = filter_table['todayHolding'].values
            rtntable.sort_index(inplace=True)
            del rtntable['rank']
        else:
            rtntable.sort_values('er', ascending=False, inplace=True)
            weights = np.zeros(total_assets)
            weights[:level] = 1.0 / level
            rtntable.loc[:, 'todayHolding'] = weights
            rtntable.sort_index(inplace=True)
        return rtntable


class ThresholdPortCalc(PortCalc):

    def __init__(self):
        pass

    def trade(self, er_table: pd.DataFrame, pre_holding: pd.DataFrame) -> pd.DataFrame:
        rtntable = er_table.copy(deep=True)

        total_assets = len(rtntable)
        level = np.percentile(rtntable['er'], 100. * (total_assets - 100) / total_assets)

        if not pre_holding.empty:
            # match pre_holding to today's codes
            rtntable.loc[:, 'todayHolding'] = 0.
            rtntable['todayHolding'] = pre_holding['todayHolding']

            sell_candiates = np.array((rtntable['er'] <= 0.) & (rtntable['todayHolding'] > 0.))
            buy_candidates = np.array((rtntable['er'] >= level) & (rtntable['todayHolding'] == 0.))
            filter_table = rtntable[buy_candidates].sort_values('er', ascending=False)
            filter_table['todayHolding'] = 0.

            total_sell_position = np.sum(rtntable.loc[sell_candiates, 'todayHolding']) + max(1.0 - np.sum(rtntable['todayHolding']), 0.)
            if total_sell_position == 0:
                return rtntable

            rtntable.loc[sell_candiates, 'todayHolding'] = 0.
            to_buy_in = min(math.ceil(total_sell_position / 0.01), len(filter_table))
            to_buy_in_list = filter_table.index[:to_buy_in]
            filter_table.loc[to_buy_in_list, 'todayHolding'] = total_sell_position / to_buy_in

            rtntable.loc[filter_table.index, 'todayHolding'] = filter_table['todayHolding'].values
        else:
            flags = np.array(rtntable['er'] >= level)
            active_assets = np.sum(flags)
            weights = np.zeros(total_assets)
            weights[flags] = 1.0 / active_assets
            rtntable.loc[:, 'todayHolding'] = weights
        return rtntable