# -*- coding: utf-8 -*-
u"""
Created on 2016-9-2

@author: cheng.li
"""

from FactorModel.providers import Provider


class Scheduler(object):

    def __init__(self,
                 provider: Provider,
                 freq='daily'):
        self.provider = provider
        self.freq = freq

    def is_rebalance(self, date):
        if self.freq == 'daily':
            return True
        elif self.freq == 'weekly':
            try:
                return self.provider.date_table.loc[date, 'eow'] == 1
            except KeyError:
                return False
        elif self.freq == 'monthly':
            try:
                return self.provider.date_table.loc[date, 'eom'] == 1
            except KeyError:
                return False