# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import abc
import copy
from typing import List
from typing import Optional
import pandas as pd
from FactorModel.utilities import load_mat


class Provider(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __iter__(self):
        pass


class FileProvider(Provider):

    def __init__(self, file_path: str, rows: Optional[int]=None):
        self.repository = load_mat(file_path, rows)
        self.calc_date_list = self.repository.calcDate.unique()
        self.apply_date_list = self.repository.applyDate.unique()

    @property
    def source_data(self) -> pd.DataFrame:
        return self.repository.copy(deep=True)

    def calc_dates(self) -> List[pd.Timestamp]:
        return copy.deepcopy(self.calc_date_list)

    def apply_dates(self) -> List[pd.Timestamp]:
        return copy.deepcopy(self.apply_date_list)

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
            raise ValueError("{date_type} is not recognized."
                             .format(date_type=date_type))

        if fields:
            return self.repository.loc[flags, fields]
        else:
            return self.repository.loc[flags, :]

    def __iter__(self):
        for calc_date, apply_date in zip(self.calc_date_list, self.apply_date_list):
            yield calc_date, apply_date, self.fetch_values_from_repo(apply_date)


