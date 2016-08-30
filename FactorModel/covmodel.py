# -*- coding: utf-8 -*-
u"""
Created on 2016-8-30

@author: cheng.li
"""

from typing import List
import numpy as np
import pandas as pd
from FactorModel.providers import Provider
from FactorModel.facts import INDUSTRY_LIST
from FactorModel.facts import STYLE_LIST


class CovModel(object):

    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    def  fetch_cov(self, calc_date: str) -> np.array:
        calc_date = pd.Timestamp(calc_date)
        corr_mat = self.provider.fetch_factor_corr(calc_date)
        factor_vol = self.provider.fetch_factor_vol(calc_date)
        sr_level = self.provider.fetch_risk_level(calc_date)
        sr_style = self.provider.fetch_risk_style(calc_date, INDUSTRY_LIST + STYLE_LIST)

        fields = ['calcDate'] + INDUSTRY_LIST + STYLE_LIST
        data = self.provider.fetch_values_from_repo(calc_date, 'calc_date', fields)

        ind_factor_loading = data[INDUSTRY_LIST].values
        style_factor_loading = data[STYLE_LIST].values

        assets_number = len(data)
        factor_loading = np.concatenate([np.ones((assets_number, 1)), ind_factor_loading, style_factor_loading], axis=1)

        res_vols = sr_level * np.exp(sr_style @ factor_loading[:, 1:].T)
        return np.diag(np.power(res_vols, 2))

if __name__ == "__main__":
    from FactorModel.providers import DBProvider
    db_provider = DBProvider('10.63.6.219', 'sa', 'A12345678!')
    db_provider.load_data('2008-01-02', '2010-02-01', ['Growth', 'CFinc1', 'Rev5m'])

    cov_model = CovModel(db_provider)
    cov_model.fetch_cov('2010-01-29')
