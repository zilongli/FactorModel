# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import numpy as np


class PortCalc(object):

    def __init__(self):
        pass

    def trade(self, er_table):
        total_assets = len(er_table)
        level = np.percentile(er_table['er'], 100. * (total_assets - 100) / total_assets)
        flags = np.array(er_table['er'] >= level)
        active_assets = np.sum(flags)

        weights = np.zeros(total_assets)
        weights[flags] = 1.0 / active_assets

        er_table['weight'] = weights
        return er_table


if __name__ == "__main__":
    import pandas as pd

    er = np.random.randn(800)
    codes = range(800)

    er_table = pd.DataFrame(er, index=codes, columns=['er'])
    port_calc = PortCalc()
    port_calc.trade(er_table)
    print(er_table)
