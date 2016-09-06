# -*- coding: utf-8 -*-
u"""
Created on 2016-9-5

@author: cheng.li
"""
import datetime as dt
import pandas as pd
import numpy as np
from pandas.io.excel import ExcelWriter
from FactorModel.performance import PerfAttributeLOO
from FactorModel.performance import PerfAttributeAOI
from FactorModel.portcalc import ERRankPortCalc
from FactorModel.schedule import Scheduler
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.simulator import Simulator
from FactorModel.providers import FileProvider
from FactorModel.analysers import PnLAnalyser

try:
    import seaborn as sns
    sns.set_style('ticks')
except ImportError:
    pass

factor_names = ['Growth', 'HRL', 'R5MOHRL']
env = FileProvider("d:/data.pkl")
trainer = ERModelTrainer(250, 1, 10)
trainer.train_models(factor_names, env.source_data)#, train_dates=['2010-02-01'])
cov_model = CovModel(env)
port_calc = ERRankPortCalc(100, 101)
scheduler = Scheduler(env, 'daily')
simulator = Simulator(env, trainer, cov_model, scheduler, port_calc)
analyser = PnLAnalyser()

df1 = env.source_data
df2 = simulator.simulate()

df1 = df1.loc[df2.index[0]:, :]
df1[df2.columns] = df2

from matplotlib import pyplot as plt

attributer = PerfAttributeLOO()
attributer.analysis(trainer, scheduler, port_calc, df1)

attributer2 = PerfAttributeAOI()
attributer2.analysis(trainer, scheduler, port_calc, df1)

attributer.plot()
plt.title('Performance Attribution (Leave One Out)')

attributer2.plot()
plt.title('Performance Attribution (Add One In)')

plt.show()

with ExcelWriter('result2.xlsx') as wb:
    attributer.report.to_excel(wb, 'loo')
    attributer2.report.to_excel(wb, 'aoi')
