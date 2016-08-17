# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

from typing import List
from typing import Optional
import pandas as pd


class Env(object):

    def __init__(self, repository: pd.DataFrame) -> None:
        self.repository = repository

    def calc_dates(self) -> List[pd.Timestamp]:
        return self.repository.calcDate.unique()

    def apply_dates(self) -> List[pd.Timestamp]:
        return self.repository.applyDate.unique()

    def fetch_values_from_repo(self, date: pd.Timestamp,
                               date_type: Optional[str]='apply_date',
                               fields: Optional[List[str]]=None) -> pd.DataFrame:
        if date_type.lower() == 'apply_date':
            if fields:
                return self.repository.loc[date, fields]
            else:
                return self.repository.loc[date, :]

        elif date_type.lower() == 'calc_date':
            flags = self.repository.calcDate == date
        else:
            raise ValueError("{date_type} is not recognized.".format(date_type=date_type))

        if fields:
            return self.repository.loc[flags, fields]
        else:
            return self.repository.loc[flags, :]
