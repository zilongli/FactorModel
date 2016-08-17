# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import bisect
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day


class ERModelTrainer(object):

    def __init__(self, win_size, periods, decay):
        self.win_size = win_size
        self.periods = periods
        self.decay = decay
        self.models = None
        self.yield_name = 'D' + str(self.decay) + 'Res'

    def fetch_model(self, date):
        i = bisect.bisect_left(self.models.index, date)
        if self.models.index[i] == date:
            return self.models.iloc[i, :]
        else:
            if i - 1 >= 0:
                return self.models.iloc[i-1, :]
            else:
                return pd.Series()

    def train_models(self, factors, train_data):
        apply_dates = train_data.applyDate.unique()
        calc_dates = train_data.calcDate.unique()
        model_data = self._calc_model_dates(apply_dates, calc_dates)

        train_data = train_data[[self.yield_name, *factors]]
        time_line = train_data.index
        train_data = train_data.as_matrix()

        model_data['model'] = None
        for i in range(len(model_data)):
            dates = model_data.iloc[i, :]
            model = self._train(dates, time_line, train_data)
            model_data.loc[dates.name, 'model'] = model
        self.models = model_data

    def _train(self, model_dates, time_line, train_data):
        train_start_date = model_dates.trainStart
        train_end_date = model_dates.trainEnd

        left = bisect.bisect_left(time_line, train_start_date)
        right = bisect.bisect_left(time_line, train_end_date)

        Y = train_data[left:right, 0]
        X = train_data[left:right, 1:]

        model = ERModel()
        model.fit(X, Y)
        return model

    def _calc_model_dates(self, apply_dates, calc_dates):
        model_dates = []
        model_calc_dates = []
        train_start_dates = []
        train_end_dates = []

        for i, apply_date in enumerate(apply_dates):
            factor_end = i - self.decay
            factor_start = i - self.win_size - self.decay
            if factor_start >= 0 and factor_start % self.periods == 0:
                model_dates.append(apply_date)
                model_calc_dates.append(calc_dates[i])
                train_start_dates.append(apply_dates[factor_start])
                train_end_dates.append(apply_dates[factor_end])

        data = np.array([model_calc_dates, train_start_dates, train_end_dates]).T
        return pd.DataFrame(data=data,
                            columns=['calcDate', 'trainStart', 'trainEnd'],
                            index=model_dates)


class ERModel(object):

    def __init__(self, model_type='ols'):
        self.model_type = model_type
        self.model_params = None

    def fit(self, X, Y):
        if self.model_type.lower() == 'ols':
            self.model_params = np.linalg.lstsq(X, Y)[0]

    def calculate_er(self, factor_values):
        return factor_values @ self.model_params


if __name__ == "__main__":
    from utilities import load_mat
    from Env import Env
    import mkl
    mkl.set_num_threads(4)
    df = load_mat("d:/data.mat")
    calc_dates = df.calcDate.unique()
    apply_dates = df.applyDate.unique()

    model_trainer = ERModelTrainer(250, 1, 10)
    model_trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], df)
