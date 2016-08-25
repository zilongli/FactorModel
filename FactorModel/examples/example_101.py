# -*- coding: utf-8 -*-
u"""
Created on 2016-8-25

@author: cheng.li
"""

from FactorModel.portcalc import MeanVariancePortCalc
from FactorModel.ermodel import ERModelTrainer
from FactorModel.simulator import Simulator
from FactorModel.providers import FileProvider

import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('ticks')

env = FileProvider("/home/wegamekinglc/Downloads/data.mat", rows=None)
trainer = ERModelTrainer(250, 1, 10)
trainer.train_models(['Growth', 'CFinc1', 'Rev5m'], env.source_data)
port_calc = MeanVariancePortCalc('cost_budget', 2e-4)
simulator = Simulator(env, trainer, port_calc)
df = simulator.simulate()

analyser = PnLAnalyser()

raw_data = env.source_data
raw_data = raw_data.set_index('code', append=True)
df = df.set_index('code', append=True)
raw_data[df.columns] = df
raw_data.dropna(inplace=True)
returns = analyser.calculate(raw_data)
analyser.plot()
plt.show()
