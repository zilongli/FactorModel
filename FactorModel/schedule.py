# -*- coding: utf-8 -*-
u"""
Created on 2016-9-2

@author: cheng.li
"""

import numpy as np
from FactorModel.providers import Provider


class Scheduler(object):

    def __init__(self,
                 provider: Provider,
                 freq='daily',
                 start_date=None):
        self.date_table = provider.date_table.copy(deep=True)
        self.freq = freq
        if start_date:
            self.start_date = np.datetime64(start_date)
        else:
            self.start_date = np.datetiem64('1970-01-01')

        if self.freq == 'biweekly':
            week_ends = \
                self.date_table['eow'][self.date_table.eow == 1].astype(int)
            week_ends[:] = 0
            week_ends[::2] = 1
            self.date_table['beow'] = week_ends
            self.date_table.fillna(0, inplace=True)

    def is_rebalance(self, date):
        if date >= self.start_date:
            if self.freq == 'daily':
                return True
            elif self.freq == 'weekly':
                try:
                    return self.date_table.loc[date, 'eow'] == 1
                except KeyError:
                    return False
            elif self.freq == 'monthly':
                try:
                    return self.date_table.loc[date, 'eom'] == 1
                except KeyError:
                    return False
            elif self.freq == 'biweekly':
                try:
                    return self.date_table.loc[date, 'beow'] == 1
                except KeyError:
                    return False
        else:
            return False
