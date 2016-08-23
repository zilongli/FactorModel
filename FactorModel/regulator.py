# -*- coding: utf-8 -*-
u"""
Created on 2016-8-22

@author: cheng.li
"""

from typing import Tuple
from typing import Any
from typing import List
import numpy as np
import pandas as pd
from collections import namedtuple


Constraints = namedtuple('Constraints', ['lb', 'ub', 'lc', 'lct'])


class Regulator(object):

    def __init__(self, industry_list: List[str]) -> None:
        self.industry_list = industry_list

    def build_constraints(self, data: pd.DataFrame) -> Tuple[Constraints, Constraints]:
        industry_factor = data[self.industry_list].values
        assets_number = len(data)

        benchmark_weights = data['zz500'].values
        benchmark_weights.shape = -1, 1

        equality_cons = np.ones((1, assets_number))
        equality_cons = np.concatenate((equality_cons, industry_factor.T), axis=0)
        equality_value = np.zeros((np.size(equality_cons, 0), 1))
        equality_value +=  equality_cons @ benchmark_weights
        equality_cons = np.concatenate((equality_cons, equality_value), axis=1)
        equality_cons = equality_cons[:-1, :]

        lb = np.zeros(assets_number)
        ub = np.ones(assets_number) * 0.02 + benchmark_weights

        lc = equality_cons
        lct = np.zeros(np.size(equality_cons, 0))

        tading_constrains = Constraints(lb=lb, ub=ub, lc=lc, lct=lct)
        regulator_constrains = Constraints(lb=lb, ub=ub, lc=lc, lct=lct)
        return tading_constrains, regulator_constrains


if __name__ == "__main__":
    from FactorModel.utilities import load_mat
    from FactorModel.facts import INDUSTRY_LIST
    df = load_mat("d:/data.mat", rows=20000)
    data = df.loc['2008-01-03', :]

    reg = Regulator(INDUSTRY_LIST)
    reg.build_constraints(data)
