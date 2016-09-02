# -*- coding: utf-8 -*-
u"""
Created on 2016-8-30

@author: cheng.li
"""

import numpy as np
import pandas as pd
from FactorModel.providers import Provider
from FactorModel.facts import INDUSTRY_LIST
from FactorModel.facts import STYLE_LIST


class CovModel(object):

    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    def fetch_cov(self, calc_date: str, repo_data: pd.DataFrame) -> np.array:
        calc_date = pd.Timestamp(calc_date)
        try:
            sr_level = self.provider.fetch_risk_level(calc_date)
            sr_style = self.provider.fetch_risk_style(calc_date, INDUSTRY_LIST + STYLE_LIST)
        except TypeError:
            return np.array([])

        ind_factor_loading = repo_data[INDUSTRY_LIST].values
        style_factor_loading = repo_data[STYLE_LIST].values

        assets_number = len(repo_data)
        factor_loading = np.concatenate([np.ones((assets_number, 1)), ind_factor_loading, style_factor_loading], axis=1)

        res_vols = sr_level * np.exp(sr_style @ factor_loading[:, 1:].T)
        return np.diag(np.power(res_vols, 2))
