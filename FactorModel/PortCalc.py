# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import pandas as pd
import numpy as np


class PortCalc(object):

    def __init__(self):
        pass

    def trade(self, er_table: pd.DataFrame) -> pd.DataFrame:
        rtntable = er_table.copy(deep=True)

        total_assets = len(rtntable)
        level = np.percentile(rtntable['er'], 100. * (total_assets - 100) / total_assets)
        flags = np.array(rtntable['er'] >= level)
        active_assets = np.sum(flags)

        weights = np.zeros(total_assets)
        weights[flags] = 1.0 / active_assets

        rtntable['weight'] = weights
        return rtntable


if __name__ == "__main__":

    er = np.random.randn(800)
    codes = range(800)

    er_table = pd.DataFrame(er, index=codes, columns=['er'])
    port_calc = PortCalc()
    port_calc.trade(er_table)
    print(er_table)
