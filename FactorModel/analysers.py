# -*- coding: utf-8 -*-
u"""
Created on 2016-8-17

@author: cheng.li
"""

import numpy as np
import pandas as pd


class PnLAnalyser(object):

    def __init__(self) -> None:
        self.report = None

    def calculate(self,
                  data: pd.DataFrame) -> pd.DataFrame:
        all_dates = np.unique(data.index)
        everyday_return = []
        everyday_return_after_tc = []
        everyday_tc = []
        for date in all_dates:
            returns = data.loc[date, ['nextReturn1day', 'evolvedBMWeight']]
            weights = data.loc[date, 'todayHolding']
            pre_weights = data.loc[date, 'preHolding']
            this_day_return = returns['nextReturn1day'].values.T @ (weights.values - returns['evolvedBMWeight'].values)
            this_day_tc = np.sum(np.abs(weights.values - pre_weights.values)) * 0.002
            everyday_return.append(this_day_return)
            everyday_tc.append(this_day_tc)
            everyday_return_after_tc.append(this_day_return - this_day_tc)
        return_table = pd.DataFrame(data=np.array([everyday_return, everyday_return_after_tc, everyday_tc]).T,
                                    index=all_dates,
                                    columns=['pnl', 'pnl - tc', 'tc'])
        self.report = return_table
        return return_table.copy()

    def plot(self) -> None:
        self.report.cumsum().plot()
