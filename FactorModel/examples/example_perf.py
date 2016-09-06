# -*- coding: utf-8 -*-
u"""
Created on 2016-9-5

@author: cheng.li
"""

from pandas.io.excel import ExcelWriter
from matplotlib import pyplot as plt
from FactorModel.performance import PerfAttributeLOO
from FactorModel.performance import PerfAttributeAOI
from FactorModel.performance import PerfAttributeFocusLOO
from FactorModel.performance import PerfAttributeFocusAOI
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
scheduler = Scheduler(env, 'weekly')
simulator = Simulator(env, trainer, cov_model, scheduler, port_calc)
analyser = PnLAnalyser()

df1 = env.source_data
df2 = simulator.simulate()

df1 = df1.loc[df2.index[0]:, :]
df1[df2.columns] = df2

with ExcelWriter('result_weekly.xlsx') as f:
    for cls in [PerfAttributeLOO, PerfAttributeAOI, PerfAttributeFocusLOO, PerfAttributeFocusAOI]:
        cls_name = cls.__name__
        attributer = cls()
        attributer.analysis(trainer, scheduler, port_calc, df1)
        attributer.plot()
        plt.title('Performance Attribution ' + '(' + cls_name + ')')
        attributer.report.to_excel(f, cls_name)

plt.show()
