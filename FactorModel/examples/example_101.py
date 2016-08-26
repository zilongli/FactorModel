# -*- coding: utf-8 -*-
u"""
Created on 2016-8-25

@author: cheng.li
"""

from FactorModel.utilities import combine
from FactorModel.portcalc import MeanVariancePortCalc
from FactorModel.ermodel import ERModelTrainer
from FactorModel.simulator import Simulator
from FactorModel.providers import FileProvider
from FactorModel.analysers import PnLAnalyser

import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('ticks')

env = FileProvider("/home/wegamekinglc/Downloads/data.mat", rows=None)
trainer = ERModelTrainer(250, 1, 10)
trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], env.source_data)
port_calc = MeanVariancePortCalc('cost_budget', 2e-4)
simulator = Simulator(env, trainer, port_calc)
analyser = PnLAnalyser()

df1 = env.source_data
df2 = simulator.simulate()

raw_data = combine(df1, df2)

returns = analyser.calculate(raw_data)
analyser.plot()
plt.show()
