# -*- coding: utf-8 -*-
u"""
Created on 2016-8-29

@author: cheng.li
"""
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

factor_names = ['Growth', 'HRL', 'R5MOHRL']
env = DBProvider(
    'rm-bp1jv5xy8o62h2331o.sqlserver.rds.aliyuncs.com:3433',
    'wegamekinglc',
    'We051253524522')
env.load_data('2008-01-02', '2015-11-01', factor_names)
trainer = ERModelTrainer(250, 1, 10)
trainer.train_models(factor_names, env.source_data)
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
plt.show()

env.archive('~/Downloads/data.pkl')
