# -*- coding: utf-8 -*-
u"""
Created on 2016-8-29

@author: cheng.li
"""
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

env = DBProvider('rm-bp1jv5xy8o62h2331o.sqlserver.rds.aliyuncs.com:3433', 'wegamekinglc', 'We051253524522')
env.load_data('2008-01-02', '2015-11-01', ['Growth', 'CFinc1', 'Rev5m'])
trainer = ERModelTrainer(250, 1, 10)
trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], env.source_data)
cov_model = CovModel(env)
port_calc = ERRankPortCalc()
simulator = Simulator(env, trainer, cov_model, port_calc)
analyser = PnLAnalyser()

df1 = env.source_data
df2 = simulator.simulate()

raw_data = combine(df1, df2)

returns = analyser.calculate(raw_data)
analyser.plot()
plt.show()
