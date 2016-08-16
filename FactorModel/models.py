# -*- coding: utf-8 -*-
u"""
Created on 2016-5-23

@author: cheng.li
"""

import numpy as np
import pandas as pd


class ERModel:

    def __init__(self, factor_weights):
        self.factor_weights = factor_weights

    def calc_expected_return(self, factor_values):
        merged_df = pd.merge(self.factor_weights, factor_values, on='code')
        return np.dot(merged_df['weight'], merged_df['factor'])


if __name__ == "__main__":
    factor_weights = pd.DataFrame({'code': ['000001', '000002', '000003'],
                                   'weight': [0.3, 0.3, 0.4]})
    factor_values = pd.DataFrame({'code': ['000001', '000003', '000002'],
                                  'factor': [1.0, 2.0, 3.0]})

    model = ERModel(factor_weights)
    print(model.calc_expected_return(factor_values))
