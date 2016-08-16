# -*- coding: utf-8 -*-
u"""
Created on 2016-5-23

@author: cheng.li
"""

from FactorModel.utilities import create_factor_list
from FactorModel.utilities import mf_config
from FactorModel.utilities import pm_config

FUNDAMENTAL_FACTOR = create_factor_list(mf_config, 'FactorData')[5:]

TA_FACTOR = create_factor_list(pm_config, 'AlphaFactors_Licheng')[2:]
TA_FACTOR = TA_FACTOR[~(TA_FACTOR.str.endswith('initsellamt'))]
TA_FACTOR = TA_FACTOR[~(TA_FACTOR.str.endswith('_diff'))]

RISK_FACTOR = create_factor_list(pm_config, 'RiskFactors')[2:]

MONEY_FACTOR = create_factor_list(mf_config, 'WindMoneyFlow1')[2:]
MONEY_FACTOR = MONEY_FACTOR[MONEY_FACTOR != 'BAR']

INDUSTRY_LIST = ('CommunicationsAndTransportation',
                 'LeisureServices',
                 'MultiMedia',
                 'PublicUtility',
                 'Agriculture',
                 'ChemicalIndustry',
                 'MedicationAndBio',
                 'CommercialTrade',
                 'DefenseIndustry',
                 'HouseholdAppliances',
                 'ConstructionAndMaterial',
                 'BuildingDecoration',
                 'RealEstate',
                 'DiversifiedMetal',
                 'Machinary',
                 'MotorVehicle',
                 'ElectronicIndustry',
                 'ElectricalEquip',
                 'TextileAndGarment',
                 'Synthetics',
                 'Computer',
                 'LightManufacturing',
                 'Telecoms',
                 'ExtractiveIndustry',
                 'Metal',
                 'Bank',
                 'NonBankFinancial',
                 'FoodAndBeverage')
