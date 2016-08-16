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
        level = np.percentile(er_table['er'], 100. * (len(er_table)  - 100) / len(er_table))
        er_table['weight'] = 0.
        flags = er_table['er'] >= level
        active_assets = sum(flags)
        er_table.loc[er_table['er'] >= level, 'weight'] = 1.0 / active_assets
        return er_table


if __name__ == "__main__":

    import numpy as np
    import pandas as pd

    er = np.random.randn(800)
    codes = range(800)

    er_table = pd.DataFrame(er, index=codes, columns=['er'])
    port_calc = PortCalc()
    port_calc.trade(er_table)
    print(er_table)
