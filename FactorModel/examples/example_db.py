# -*- coding: utf-8 -*-
u"""
Created on 2016-8-29

@author: cheng.li
"""
import datetime as dt
from FactorModel.utilities import combine
from FactorModel.portcalc import ERRankPortCalc
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.simulator import Simulator
from FactorModel.providers import DBProvider
from FactorModel.analysers import PnLAnalyser

import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('ticks')

start = dt.datetime.now()
env = DBProvider('10.63.6.219', 'sa', 'A12345678!')
env.load_data('2008-01-02', '2012-11-01', ['Growth', 'CFinc1', 'Rev5m'])
trainer = ERModelTrainer(250, 1, 10)
trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], env.source_data)
cov_model = CovModel(env)
port_calc = ERRankPortCalc()
simulator = Simulator(env, trainer, cov_model, port_calc)
analyser = PnLAnalyser()

print(dt.datetime.now() - start)
start = dt.datetime.now()

df1 = env.source_data
df2 = simulator.simulate()

print(dt.datetime.now() - start)
start = dt.datetime.now()

df1 = df1.loc[df2.index[0]:, :]
df1[df2.columns] = df2

print(dt.datetime.now() - start)
start = dt.datetime.now()

returns = analyser.calculate(df1)
analyser.plot()
print(dt.datetime.now() - start)
plt.show()
