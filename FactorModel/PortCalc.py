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

            sell_candiates = np.array((rtntable['er'] <= 0.) & (rtntable['todayHolding'] > 0.))
            buy_candidates = np.array((rtntable['er'] >= level) & (rtntable['todayHolding'] == 0.))
            filter_table = rtntable[buy_candidates].sort_values('er', ascending=False)
            filter_table['todayHolding'] = 0.

            total_sell_position = np.sum(rtntable.loc[sell_candiates, 'todayHolding']) + max(1.0 - np.sum(rtntable['todayHolding']), 0.)
            if total_sell_position == 0:
                return rtntable

            rtntable.loc[sell_candiates, 'todayHolding'] = 0.
            to_buy_in = min(math.ceil(total_sell_position / 0.01), len(filter_table))

            filter_table['todayHolding'][0:to_buy_in] = total_sell_position / to_buy_in

            rtntable.loc[filter_table.index, 'todayHolding'] = filter_table['todayHolding'].values
        else:
            flags = np.array(rtntable['er'] >= level)
            active_assets = np.sum(flags)
            weights = np.zeros(total_assets)
            weights[flags] = 1.0 / active_assets
            rtntable['todayHolding'] = weights
        return rtntable
