u"""
Created on 2016-9-12

@author: cheng.li
"""

import numpy as np
from FactorModel.patterns import Singleton
from FactorModel.parameters import create_risk_aversion


class SettingsFactory(metaclass=Singleton):

    def __init__(self):

        self.ra_calc = create_risk_aversion('percent', 1)

    def risk_aversion(self,
                      er: np.array,
                      cov: np.array) -> float:
        return self.ra_calc(er, cov)


Settings = SettingsFactory()
