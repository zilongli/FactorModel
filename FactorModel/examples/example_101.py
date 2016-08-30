# -*- coding: utf-8 -*-
u"""
Created on 2016-8-25

@author: cheng.li
"""

from FactorModel.utilities import combine
from FactorModel.portcalc import MeanVariancePortCalc
from FactorModel.ermodel import ERModelTrainer
from FactorModel.simulator import Simulator
from FactorModel.providers import FileProvider
from FactorModel.analysers import PnLAnalyser

import seaborn as sns
from matplotlib import pyplot as plt

import datetime as dt

sns.set_style('ticks')

start = dt.datetime.now()

env = FileProvider("/home/wegamekinglc/Downloads/data.mat", rows=None)
trainer = ERModelTrainer(250, 1, 10)
trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], env.source_data)
port_calc = MeanVariancePortCalc('cost_budget', 2e-4)
simulator = Simulator(env, trainer, port_calc)
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

returns = analyser.calculate(df2)
analyser.plot()
print(dt.datetime.now() - start)
plt.show()
