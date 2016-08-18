# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

from typing import List
from typing import Any
import pandas as pd
from FactorModel.Env import Env
from FactorModel.ERModel import ERModelTrainer
from FactorModel.PortCalc import PortCalc


class Simulator(object):

    def __init__(self, env: Env, model_factory: ERModelTrainer, port_calc: PortCalc):
        self.model_factory = model_factory
        self.env = env
        self.port_calc = port_calc
        self.info_keeper = InfoKeeper()

    def simulate(self) -> pd.DataFrame:
        apply_dates = self.env.apply_dates()
        calc_dates = self.env.calc_dates()

        pre_holding = pd.DataFrame()

        for i, apply_date in enumerate(apply_dates):
            this_data = self.env.fetch_values_from_repo(apply_date)
            codes = this_data.code.astype(int)
            model = self.model_factory.fetch_model(apply_date)
            if not model.empty:
                factor_values = this_data[['Growth', 'CFinc1', 'Rev5m']].as_matrix()
                er = model.model.calculate_er(factor_values)
                er_table = pd.DataFrame(er, index=codes, columns=['er'])
                positions = self.port_calc.trade(er_table, pre_holding)
                if not pre_holding.empty:
                    positions['preHolding'] = pre_holding['todayHolding']
                    positions.fillna(0., inplace=True)
                else:
                    positions['preHolding'] = 0.0
                self.log_info(apply_date, calc_dates[i], positions)

                pre_holding = positions[['todayHolding']]

        return self.info_keeper.view()

    def log_info(self, apply_date: pd.Timestamp, calc_date: pd.Timestamp, positions: pd.DataFrame) -> None:
        codes = positions.index
        apply_dates = [apply_date] * len(codes)
        calc_dates = [calc_date] * len(codes)
        self.info_keeper.attach_list(apply_dates, codes, 'calcDate', calc_dates)
        for col in positions:
            to_store = positions[col]
            self.info_keeper.attach_list(apply_dates, codes, col, list(to_store.values))


class InfoKeeper(object):

    def __init__(self):
        self.info = {}
        self.labels = []

    def attach(self, datetime: pd.Timestamp, code: List[int], label: str, value: Any) -> None:
        if label not in self.info:
            self.info[label] = ([], [], [])
            self.labels.append(label)

        self.info[label][0].append(datetime)
        self.info[label][1].append(code)
        self.info[label][2].append(value)

    def attach_list(self, datetimes: List[pd.Timestamp], codes: List[int], label: str, values: List[Any]) -> None:
        if label not in self.info:
            self.info[label] = ([], [], [])
            self.labels.append(label)
        self.info[label][0].extend(datetimes)
        self.info[label][1].extend(codes)
        self.info[label][2].extend(values)

    def view(self) -> pd.DataFrame:
        series_list = []
        for s in self.labels:
            series = pd.Series(self.info[s][2], index=[self.info[s][0], self.info[s][1]])
            series_list.append(series)

        if series_list:
            res = pd.concat(series_list, axis=1, join='outer')
            res.set_axis(axis=1, labels=self.labels)
            res = res.reset_index(1).rename(columns={'level_1':'code'})
        else:
            res = pd.DataFrame()
        return res


if __name__ == "__main__":

    from FactorModel.utilities import load_mat
    df = load_mat("d:/data.mat", rows=220000)
    env = Env(df)
    trainer = ERModelTrainer(250, 1, 10)
    trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], df)
    port_calc = PortCalc()
    simulator = Simulator(env, trainer, port_calc)
    simulator.simulate()
    print(simulator.info_keeper.view())
