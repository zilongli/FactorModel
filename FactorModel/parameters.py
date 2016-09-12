u"""
Created on 2016-9-12

@author: cheng.li
"""

import abc
import math
import numpy as np


class RiskAversionBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self,
                 er: np.array,
                 cov: np.array) -> float:
        pass


class RARelative(RiskAversionBase):

    def __init__(self, level):
        self.level = level

    def __call__(self,
                 er: np.array,
                 cov: np.array) -> float:
        t = np.linalg.solve(cov @ cov, er)
        sqrt_root = math.sqrt(np.dot(er, t))
        return self.level * sqrt_root


def create_risk_aversion(type: str,
                         level: float) -> RiskAversionBase:
    if type == 'percent':
        return RARelative(level)
    else:
        raise ValueError('({0}) is not a valid risk aversion type.'
                         .formt(type))
