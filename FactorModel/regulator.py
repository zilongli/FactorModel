# -*- coding: utf-8 -*-
u"""
Created on 2016-8-22

@author: cheng.li
"""

from typing import Tuple
from typing import List
import numpy as np
import pandas as pd
from collections import namedtuple
from FactorModel.facts import BENCHMARK


Constraints = namedtuple('Constraints', ['lb', 'ub', 'lc', 'lct', 'suspend'])


class Regulator(object):

    def __init__(self, industry_list: List[str]) -> None:
        self.industry_list = industry_list

    def build_constraints(self, data: pd.DataFrame) \
            -> Tuple[Constraints, Constraints]:
        industry_factor = data[self.industry_list].values
        assets_number = len(data)

        benchmark_weights = data[BENCHMARK].values.copy()
        benchmark_weights.shape = -1, 1

        suspended_flags = \
            (data['Suspend20DayTrailing'] < 0.8) \
            | (data['Suspend5DayTrailing'] < 0.9)

        equality_cons = np.ones((1, assets_number))
        equality_cons = np.concatenate(
            (equality_cons, industry_factor.T),
            axis=0)
        equality_value = np.zeros((np.size(equality_cons, 0), 1))
        equality_value += equality_cons @ benchmark_weights
        equality_cons = np.concatenate((equality_cons, equality_value), axis=1)
        equality_cons = equality_cons[:-1, :]

        lb = np.zeros(assets_number)
        ub = np.ones(assets_number) * 0.02 + benchmark_weights.flatten()

        lc = equality_cons
        lct = np.zeros(np.size(equality_cons, 0))

        trading_constrains = Constraints(
            lb=lb, ub=ub, lc=lc, lct=lct, suspend=suspended_flags.values)
        regulator_constrains = Constraints(
            lb=lb, ub=ub, lc=lc, lct=lct, suspend=suspended_flags.values)
        return trading_constrains, regulator_constrains
