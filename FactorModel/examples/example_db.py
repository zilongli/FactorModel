# -*- coding: utf-8 -*-
u"""
Created on 2016-8-29

@author: cheng.li
"""
from FactorModel.portcalc import ERRankPortCalc
from FactorModel.schedule import Scheduler
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.portcalc import MeanVariancePortCalc
from FactorModel.simulator import Simulator
from FactorModel.providers import MSSQLProvider
from FactorModel.analysers import PnLAnalyser
from FactorModel.regulator import Regulator
from FactorModel.facts import INDUSTRY_LIST
from matplotlib import pyplot as plt

try:
    import seaborn as sns
    sns.set_style('ticks')
except ImportError:
    pass

factor_names = ['RMC', 'RVS', 'D5M5']
env = MSSQLProvider(
    'rm-bp1jv5xy8o62h2331o.sqlserver.rds.aliyuncs.com:3433',
    'wegamekinglc',
    'We051253524522')
env.load_data('2008-01-02', '2015-11-01', factor_names)
trainer = ERModelTrainer(250, 1, 5)
trainer.train_models(factor_names, env.source_data)
cov_model = CovModel(env)
port_calc = MeanVariancePortCalc(method='no_cost')
scheduler = Scheduler(env, 'weekly')
constrinats_builder = Regulator(INDUSTRY_LIST)
simulator = Simulator(env,
                      trainer,
                      cov_model,
                      scheduler,
                      port_calc,
                      constrinats_builder)
analyser = PnLAnalyser()

df1 = env.source_data
df2 = simulator.simulate()

df1 = df1.loc[df2.index[0]:, :]
df1[df2.columns] = df2

returns = analyser.calculate(df1)
analyser.plot()
plt.show()

env.archive('d:/data2.pkl')
