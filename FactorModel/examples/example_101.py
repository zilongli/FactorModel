# -*- coding: utf-8 -*-
u"""
Created on 2016-8-25

@author: cheng.li
"""

from FactorModel.schedule import Scheduler
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.portcalc import MeanVariancePortCalc
from FactorModel.portcalc import ERRankPortCalc
from FactorModel.simulator import Simulator
from FactorModel.providers import FileProvider
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
env = FileProvider("d:/data2.pkl")
trainer = ERModelTrainer(250, 1, 5)
trainer.train_models(factor_names, env.source_data)
cov_model = CovModel(env)
#port_calc = MeanVariancePortCalc('cost_budget', cost_budget=2e-4)
port_calc = ERRankPortCalc(100, 200)
scheduler = Scheduler(env, 'daily')
constrinats_builder = Regulator(INDUSTRY_LIST)
simulator = Simulator(env, trainer,
                      cov_model,
                      scheduler,
                      port_calc,
                      constrinats_builder)
analyser = PnLAnalyser()

df1 = env.source_data
df2 = simulator.simulate()
print(df2.head())

df1 = df1.loc[df2.index[0]:, :]
df1[df2.columns] = df2

returns = analyser.calculate(df1)
analyser.plot()
plt.show()
