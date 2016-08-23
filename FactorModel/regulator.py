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


Constraints = namedtuple('Constraints', ['lb', 'ub'])


class Regulator(object):

    def __init__(self, industry_list: List[str]) -> None:
        self.industry_list = industry_list

    def build_constraints(self, data: pd.DataFrame) -> Tuple[Constraints, Constraints]:
        industry_factor = data[self.industry_list].values
        assets_number = len(data)
        assets_number = 10
        lb = np.zeros(assets_number)
        ub = np.ones(assets_number) * 0.02

        tading_constrains = Constraints(lb=lb, ub=ub)
        regulator_constrains = Constraints(lb=lb, ub=ub)
        return tading_constrains, regulator_constrains


if __name__ == "__main__":
    pass
