# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""


class Env(object):

    def __init__(self, repository):
        self.repository = repository

    def calc_dates(self):
        return self.repository.calcDate.unique()

    def apply_dates(self):
        return self.repository.applyDate.unique()

    def fetch_values_from_repo(self, date, dateType='apply_date', fields=None):
        if dateType.lower() == 'apply_date':
            if fields:
                return self.repository.loc[date, fields]
            else:
                return self.repository.loc[date, :]

        elif dateType.lower() == 'calc_date':
            flags = self.repository.calcDate == date
        else:
            raise ValueError("{dateType} is not recognized.".format(dateType=dateType))

        if fields:
            return self.repository.loc[flags, fields]
        else:
            return self.repository.loc[flags, :]