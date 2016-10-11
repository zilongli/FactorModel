# -*- coding: utf-8 -*-
u"""
Created on 2016-8-29

@author: cheng.li
"""

from FactorModel.schedule import Scheduler
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.portcalc import MeanVariancePortCalc
from FactorModel.portcalc import ERRankPortCalc
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

factor_names = ['Growth', 'CFinc1', 'Rev5m']
env = MSSQLProvider(
    '10.63.6.219',
    'sa',
    'A12345678!')
stock_universe = 'zz500+'
benchmark_name = 'zz500'
env.load_data('2008-01-02', '2015-11-01', factor_names, stock_universe, benchmark_name)
trainer = ERModelTrainer(250, 1, 5)
trainer.train_models(factor_names, env.source_data)
cov_model = CovModel(env)
scheduler = Scheduler(env, 'weekly')
constraints_builder = Regulator(INDUSTRY_LIST)
port_calc = MeanVariancePortCalc(method='cost_budget',
                                 model_factory=trainer,
                                 cov_model=cov_model,
                                 constraints_builder=constraints_builder,
                                 scheduler=scheduler,
                                 cost_budget=2e-4)
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

env.archive('d:/data2.pkl')
