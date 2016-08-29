# -*- coding: utf-8 -*-
u"""
Created on 2016-8-29

@author: cheng.li
"""
from FactorModel.utilities import combine
from FactorModel.portcalc import MeanVariancePortCalc
from FactorModel.ermodel import ERModelTrainer
from FactorModel.simulator import Simulator
from FactorModel.providers import DBProvider
from FactorModel.analysers import PnLAnalyser

import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('ticks')

env = DBProvider('10.63.6.219', 'sa', 'A12345678!')
env.fetch_data(20080102, 20090301, ['Growth', 'CFinc1', 'Rev5m'])
trainer = ERModelTrainer(250, 1, 10)
trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], env.source_data)
port_calc = ERRankPortCalc()
simulator = Simulator(env, trainer, port_calc)
analyser = PnLAnalyser()

df1 = env.source_data
df2 = simulator.simulate()

raw_data = combine(df1, df2)

returns = analyser.calculate(raw_data)
analyser.plot()
plt.show()
