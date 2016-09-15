# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import bisect
from typing import List
from typing import Optional
import numpy as np
import pandas as pd


class ERModel(object):

    def __init__(self, model_type: Optional[str] = 'ols') -> None:
        self.model_type = model_type
        self.model_params = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.model_type.lower() == 'ols':
            self.model_params = np.linalg.lstsq(x, y)[0]

    def calculate_er(self, factor_values: np.ndarray) -> np.ndarray:
        return factor_values @ self.model_params

    def __str__(self):
        return str(self.model_params)


class ERModelTrainer(object):

    def __init__(self, win_size: int, periods: int, decay: int) -> None:
        self.win_size = win_size
        self.periods = periods
        self.decay = decay
        self.models = None
        self.factor_names = None
        self.yield_name = 'D' + str(self.decay) + 'Res'

    def fetch_model(self,
                    date: pd.Timestamp) -> pd.Series:
        i = bisect.bisect_left(self.models.index, date)
        if i < len(self.models.index) and self.models.index[i] == date:
            return self.models.iloc[i, :]
        else:
            if i - 1 >= 0:
                return self.models.iloc[i-1, :]
            else:
                return pd.Series()

    def train_models(self,
                     factors: List[str],
                     train_data: pd.DataFrame,
                     train_dates: List[pd.Timestamp]=None) -> None:
        self.factor_names = factors
        apply_dates = list(train_data.applyDate.unique())
        calc_dates = list(train_data.calcDate.unique())

        model_data = self._calc_model_dates(
            train_dates,
            apply_dates,
            calc_dates)

        self._normalize(train_data, self.yield_name)

        train_data = train_data[[self.yield_name + '_norm', *factors]]

        model_data['model'] = None
        for i in range(len(model_data)):
            dates = model_data.iloc[i, :]
            model = self._train(dates, train_data)
            model_data.loc[dates.name, 'model'] = model
        self.models = model_data

    def _normalize(self, train_data, field):
        res = train_data[['applyDate', field]] \
            .groupby('applyDate') \
            .transform(lambda x: x / x.std())
        train_data[field + '_norm'] = res[field]

    def _train(self,
               model_dates: pd.Series,
               train_data: pd.DataFrame) -> ERModel:
        train_start_date = model_dates.trainStart
        train_end_date = model_dates.trainEnd

        time_line = train_data.index
        train_data = train_data.as_matrix()

        left = bisect.bisect_left(time_line, train_start_date)
        right = bisect.bisect_left(time_line, train_end_date)

        y = train_data[left:right, 0]
        x = train_data[left:right, 1:]

        model = ERModel()
        model.fit(x, y)
        return model

    def _calc_model_dates(self,
                          train_dates,
                          apply_dates: List[pd.Timestamp],
                          calc_dates: List[pd.Timestamp]) -> pd.DataFrame:
        model_dates = []
        model_calc_dates = []
        train_start_dates = []
        train_end_dates = []

        if not train_dates:
            train_dates = apply_dates
        else:
            train_dates = map(lambda x: np.datetime64(x), train_dates)

        for apply_date in train_dates:
            i = bisect.bisect_left(apply_dates, apply_date)
            factor_end = i - self.decay
            factor_start = i - self.win_size - self.decay
            if factor_start >= 0 and factor_start % self.periods == 0:
                model_dates.append(apply_date)
                model_calc_dates.append(calc_dates[i])
                train_start_dates.append(apply_dates[factor_start])
                train_end_dates.append(apply_dates[factor_end])

        data = np.array(
            [model_calc_dates,
                train_start_dates,
                train_end_dates]).T
        return pd.DataFrame(data=data,
                            columns=['calcDate', 'trainStart', 'trainEnd'],
                            index=model_dates,
                            dtype='datetime64[ns]')
