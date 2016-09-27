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
from FactorModel.portcalc import MeanVariancePortCalc
from FactorModel.portcalc import ERRankPortCalc
from FactorModel.schedule import Scheduler
from FactorModel.ermodel import ERModelTrainer
from FactorModel.covmodel import CovModel
from FactorModel.regulator import Regulator
from FactorModel.simulator import Simulator
from FactorModel.providers import FileProvider
from FactorModel.analysers import PnLAnalyser
from FactorModel.facts import INDUSTRY_LIST

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
scheduler = Scheduler(env, 'weekly')
constraints_builder = Regulator(INDUSTRY_LIST)
port_calc = MeanVariancePortCalc(method='no_cost',
                                 model_factory=trainer,
                                 cov_model=cov_model,
                                 constraints_builder=constraints_builder,
                                 scheduler=scheduler)
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

analyser.calculate(df1)
analyser.plot()
plt.show()

attributers = \
    [PerfAttributeLOO,
     PerfAttributeAOI,
     PerfAttributeFocusLOO,
     PerfAttributeFocusAOI]

weight_cols = [f + '_weight' for f in factor_names]

with ExcelWriter('result_weekly_mv2.xlsx') as f:
    for cls in attributers:
        cls_name = cls.__name__
        attributer = cls()
        attributer.analysis(trainer,
                            scheduler,
                            port_calc,
                            cov_model,
                            constraints_builder,
                            df1)
        attributer.report.to_excel(f, cls_name)
