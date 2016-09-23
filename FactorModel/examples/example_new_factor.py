# -*- coding: utf-8 -*-
u"""
Created on 2016-9-23

@author: cheng.li
"""

import pandas as pd
from FactorModel.schedule import Scheduler
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.portcalc import ERRankPortCalc
from FactorModel.portcalc import MeanVariancePortCalc
from FactorModel.providers import MSSQLProvider
from FactorModel.simulator import Simulator
from FactorModel.analysers import PnLAnalyser
from FactorModel.regulator import Regulator
from FactorModel.facts import INDUSTRY_LIST
from matplotlib import pyplot as plt

env = MSSQLProvider(
    '10.63.6.219',
    'sa',
    'A12345678!'
)

factor_name = 'factor1'

env.load_data('2014-01-01', '2015-12-31', None)

external_data = pd.read_csv('/home/wegamekinglc/Downloads/factors20160921.csv',
                            parse_dates=[0],
                            header=0,
                            names=['applyDate', 'code', factor_name])
env.append(external_data)
trainer = ERModelTrainer(250, 1, 5)
trainer.train_models([factor_name], env.source_data)
cov_model = CovModel(env)
port_calc = ERRankPortCalc(in_threshold=100, out_threshold=101)
#port_calc = MeanVariancePortCalc('cost_budget', cost_budget=2e-4)
scheduler = Scheduler(env, 'daily')
constrinats_builder = Regulator(INDUSTRY_LIST)
simulator = Simulator(env,
                      trainer,
                      cov_model,
                      scheduler,
                      port_calc,
                      constrinats_builder)

df1 = env.source_data
df2 = simulator.simulate()

df1 = df1.loc[df2.index[0]:, :]
df1[df2.columns] = df2

analyser = PnLAnalyser()
returns = analyser.calculate(df1)
analyser.plot()
plt.show()
