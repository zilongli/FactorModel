# -*- coding: utf-8 -*-
u"""
Created on 2016-9-23

@author: cheng.li
"""

import pandas as pd
from FactorModel.schedule import Scheduler
from FactorModel.ermodel import ERModelTrainer
from FactorModel.portcalc import ERRankPortCalc
from FactorModel.providers import MSSQLProvider
from FactorModel.simulator import Simulator
from FactorModel.analysers import PnLAnalyser
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('ticks')

env = MSSQLProvider(
    '10.63.6.219',
    'sa',
    'A12345678!'
)

factor_name = 'factor1'

env.load_data('2011-01-01', '2015-12-31', None, 'zz500')

external_data = pd.read_csv('H:/personal/jiangkai/factors20160927.csv',
                            parse_dates=[0],
                            header=0,
                            names=['applyDate', 'code', factor_name])
env.append(external_data)
trainer = ERModelTrainer(250, 1, 5)
trainer.train_models([factor_name], env.source_data)

scheduler = Scheduler(env, 'weekly')
port_calc = ERRankPortCalc(100,
                           101,
                           model_factory=trainer,
                           scheduler=scheduler)
simulator = Simulator(env,
                      port_calc)

df1 = env.source_data
df2 = simulator.simulate()

df1 = df1.loc[df2.index[0]:, :]
df1[df2.columns] = df2

analyser = PnLAnalyser()
returns = analyser.calculate(df1)
analyser.plot()
plt.show()
