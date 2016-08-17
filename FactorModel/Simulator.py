# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import pandas as pd


class Simulator(object):

    def __init__(self, env, model_factory, port_calc):
        self.model_factory = model_factory
        self.env = env
        self.port_calc = port_calc
        self.info_keeper = InfoKeeper()

    def simulate(self):
        apply_dates = self.env.apply_dates()
        calc_dates = self.env.calc_dates()

        for i, apply_date in enumerate(apply_dates):
            print(apply_date)
            thisData = self.env.fetch_values_from_repo(apply_date)
            codes = thisData.code.astype(int)
            model = self.model_factory.fetch_model(apply_date)
            if not model.empty:
                factor_values = thisData[['Growth', 'CFinc1', 'Rev5m']].as_matrix()
                er = model.model.calculate_er(factor_values)
                er_table = pd.DataFrame(er, index=codes, columns=['er'])
                positions = self.port_calc.trade(er_table)
                self.log_info(apply_date, calc_dates[i], positions)

    def log_info(self, apply_date, calc_date, positions):
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

    def attach(self, datetime, code, label, value):
        if label not in self.info:
            self.info[label] = ([], [], [])
            self.labels.append(label)

        self.info[label][0].append(datetime)
        self.info[label][1].append(code)
        self.info[label][2].append(value)

    def attach_list(self, datetimes, codes, label, values):
        if label not in self.info:
            self.info[label] = ([], [], [])
            self.labels.append(label)
        self.info[label][0].extend(datetimes)
        self.info[label][1].extend(codes)
        self.info[label][2].extend(values)

    def view(self):
        series_list = []
        for s in self.labels:
            series = pd.Series(self.info[s][2], index=[self.info[s][0], self.info[s][1]])
            series_list.append(series)

        if series_list:
            res = pd.concat(series_list, axis=1, join='outer')
            res.set_axis(axis=1, labels=self.labels)
        else:
            res = pd.DataFrame()
        return res


if __name__ == "__main__":
    import sys
    sys.path.append(r'D:\GitHub\wegamekinglc\FactorModels')

    from FactorModel.utilities import load_mat
    from FactorModel.Env import Env
    from FactorModel.ERModel import ERModelTrainer
    from FactorModel.PortCalc import PortCalc
    df = load_mat("d:/data.mat", rows=None)
    env = Env(df)
    trainer = ERModelTrainer(250, 1, 10)
    trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], df)
    port_calc = PortCalc()
    simulator = Simulator(env, trainer, port_calc)
    simulator.simulate()
    print(simulator.info_keeper.view())