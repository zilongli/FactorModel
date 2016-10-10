# -*- coding: utf-8 -*-
u"""
Created on 2016-8-25

@author: cheng.li
"""

from FactorModel.schedule import Scheduler
from FactorModel.ermodel import ERModelTrainer
from FactorModel.portcalc import ERRankPortCalc
from FactorModel.simulator import Simulator
from FactorModel.providers import FileProvider
from FactorModel.analysers import PnLAnalyser
from matplotlib import pyplot as plt

try:
    import seaborn as sns
    sns.set_style('ticks')
except ImportError:
    pass

factor_names = ['Growth', 'CFinc1', 'Rev5m']
env = FileProvider("d:/data2.pkl")
trainer = ERModelTrainer(250, 1, 5)
trainer.train_models(factor_names, env.source_data)
scheduler = Scheduler(env, 'weekly', '2010-02-05')
port_calc = ERRankPortCalc(100,
                           101,
                           model_factory=trainer,
                           scheduler=scheduler)
simulator = Simulator(env,
                      port_calc)
analyser = PnLAnalyser()

df1 = env.source_data
df2 = simulator.simulate()

df1 = df1.loc[df2.index[0]:, :]
df1[df2.columns] = df2

returns = analyser.calculate(df1)
analyser.plot()
plt.show()
