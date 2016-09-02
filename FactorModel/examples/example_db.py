# -*- coding: utf-8 -*-
u"""
Created on 2016-8-29

@author: cheng.li
"""
import datetime as dt
from FactorModel.portcalc import ERRankPortCalc
from FactorModel.schedule import Scheduler
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.simulator import Simulator
from FactorModel.providers import DBProvider
from FactorModel.analysers import PnLAnalyser
from matplotlib import pyplot as plt

try:
    import seaborn as sns
    sns.set_style('ticks')
except ImportError:
    pass

start = dt.datetime.now()
env = DBProvider('10.63.6.219', 'sa', 'A12345678!')
env.load_data('2008-01-02', '2011-11-01', ['Growth', 'CFinc1', 'Rev5m'])
trainer = ERModelTrainer(250, 1, 10)
trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], env.source_data)
cov_model = CovModel(env)
port_calc = ERRankPortCalc(100, 101)
scheduler = Scheduler(env, 'weekly')
simulator = Simulator(env, trainer, cov_model, scheduler, port_calc)
analyser = PnLAnalyser()

df1 = env.source_data
df2 = simulator.simulate()

df1 = df1.loc[df2.index[0]:, :]
df1[df2.columns] = df2

returns = analyser.calculate(df1)
analyser.plot()
print(dt.datetime.now() - start)
plt.show()
