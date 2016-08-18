# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import math
import pandas as pd
import numpy as np


class PortCalc(object):

    def __init__(self):
        pass

    def trade(self, er_table: pd.DataFrame, pre_holding: pd.DataFrame) -> pd.DataFrame:
        rtntable = er_table.copy(deep=True)

        total_assets = len(rtntable)
        level = np.percentile(rtntable['er'], 100. * (total_assets - 100) / total_assets)

        if not pre_holding.empty:
            # match pre_holding to today's codes
            rtntable['todayHolding'] = 0.
            rtntable['todayHolding'] = pre_holding['todayHolding']

            er = rtntable['er']
            today_holding = rtntable['todayHolding']

            sell_candiates = np.array((er <= 0.) & (today_holding > 0.))
            buy_candidates = np.array((er >= level) & (today_holding == 0.))
            filter_table = er[buy_candidates].sort_values(ascending=False)

            total_sell_position = np.sum(today_holding[sell_candiates]) + max(1. - np.sum(today_holding), 0.)
            if total_sell_position == 0:
                return rtntable

            rtntable.loc[sell_candiates, 'todayHolding'] = 0.
            to_buy_in = min(math.ceil(total_sell_position / .01), len(filter_table))
            rtntable.loc[filter_table.index[0:to_buy_in], 'todayHolding'] = total_sell_position / to_buy_in
        else:
            flags = np.array(rtntable['er'] >= level)
            active_assets = np.sum(flags)
            weights = np.zeros(total_assets)
            weights[flags] = 1.0 / active_assets
            rtntable['todayHolding'] = weights
        return rtntable
