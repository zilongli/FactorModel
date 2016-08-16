# -*- coding: utf-8 -*-
u"""
Created on 2016-5-9

@author: cheng.li
"""

import numpy as np
import pandas as pd
from FactorModel.utilities import create_conn
from FactorModel.utilities import format_codes
from FactorModel.utilities import format_date_to_index
from FactorModel.utilities import list_to_str
from FactorModel.utilities import pm_config
from FactorModel.utilities import mf_config
from FactorModel.utilities import merger
from FactorModel.facts import FUNDAMENTAL_FACTOR
from FactorModel.facts import TA_FACTOR
from FactorModel.facts import RISK_FACTOR
from FactorModel.facts import MONEY_FACTOR


def set_universe(start_date, end_date):
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')
    engine = create_conn(pm_config)
    sql = "select * from StockUniverse_500Enhance where [Date] >= '{start_date}' and [Date] <= '{end_date}'" \
        .format(start_date=start_date, end_date=end_date)

    df = pd.read_sql(sql, engine)
    df.rename(columns={'Date': 'date', 'Code': 'code'}, inplace=True)
    df = format_codes(df)
    df = format_date_to_index(df, 'date')
    return df


def fetch_data_from_table(config, universe, factors, start_date, end_date, table, to_drop=None):
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')

    engine = create_conn(config)

    fields = list_to_str(factors)
    sql = u"select [Date], [Code], {fields} from {table} " \
          u"where [Date] >= '{start_date}' and [Date] <= '{end_date}'" \
        .format(fields=fields, start_date=start_date, end_date=end_date, table=table)
    df = pd.read_sql(sql, engine)
    df.rename(columns={'Date': 'date', 'Code': 'code'}, inplace=True)

    if to_drop:
        df[df == to_drop] = np.nan
        df.dropna(inplace=True)

    df = format_date_to_index(df, 'date')
    df = format_codes(df)

    return df


def fetch_fundamental_factors(universe, factors, start_date, end_date):
    return fetch_data_from_table(mf_config,
                                 universe,
                                 factors,
                                 start_date,
                                 end_date,
                                 'FactorData')


def fetch_ta_factors(universe, factors, start_date, end_date):
    return fetch_data_from_table(pm_config,
                                 universe,
                                 factors,
                                 start_date,
                                 end_date,
                                 'AlphaFactors_Licheng')


def fetch_money_flow_factors(universe, factors, start_date, end_date):
    return fetch_data_from_table(mf_config,
                                 universe,
                                 factors,
                                 start_date,
                                 end_date,
                                 'WindMoneyFlow1',
                                 to_drop=-8888.88)


def cal_factor_chg(df, factors):
    factors = list(factors)
    dates = sorted(df.index.unique())
    previous_dates = pd.DataFrame({'date': dates, 'previous': [np.nan, *dates[:-1]]})
    flat_table = df[['code'] + factors].reset_index()
    flat_table = pd.merge(flat_table, previous_dates, on='date', how='left')
    long_previous_dates = flat_table[['code', 'previous']]
    next_previous_factors = pd.merge(long_previous_dates, flat_table, how='left', left_on=[
        'code', 'previous'], right_on=['code', 'date'])
    df[factors] = df[factors] - next_previous_factors[factors].values
    df.dropna(inplace=True)
    return df


def fetch_factor_from_all_tables(universe, factors, start_date, end_date, cal_chg=True):
    fundamental_factors = list(
        set(factors).intersection(set(FUNDAMENTAL_FACTOR)))
    technical_factors = list(set(factors).intersection(set(TA_FACTOR)))
    money_flow_factors = list(set(factors).intersection(set(MONEY_FACTOR)))

    if fundamental_factors:
        f_factor_data = fetch_fundamental_factors(
            universe, fundamental_factors, start_date, end_date)
    if technical_factors:
        ta_factor_data = fetch_ta_factors(
            universe, technical_factors, start_date, end_date)
    if money_flow_factors:
        mo_factor_data = fetch_money_flow_factors(
            universe, money_flow_factors, start_date, end_date)

    df = pd.DataFrame()
    if fundamental_factors and technical_factors:
        assert (len(f_factor_data) == len(ta_factor_data))
        df = pd.concat([f_factor_data, ta_factor_data.ix[:, ta_factor_data.columns != 'code']], axis=1)
    elif fundamental_factors:
        df = f_factor_data
    elif technical_factors:
        df = ta_factor_data
    elif money_flow_factors:
        df = mo_factor_data

    if cal_chg:
        df = cal_factor_chg(df, factors)

    return df


def fetch_trading_info(universe, factors, start_date, end_date):
    return fetch_data_from_table(mf_config,
                                 universe,
                                 factors,
                                 start_date,
                                 end_date,
                                 'TradingInfo1')


def fetch_risk_factors(universe, factors, start_date, end_date):
    return fetch_data_from_table(pm_config,
                                 universe,
                                 factors,
                                 start_date,
                                 end_date,
                                 'RiskFactors')


def fetch_index_components(universe, factors, start_date, end_date):
    return fetch_data_from_table(mf_config,
                                 universe,
                                 factors,
                                 start_date,
                                 end_date,
                                 'IndexComponents')


def attach_index_components(universe, df_factor, start_date, end_date,):
    df_index = fetch_index_components(
        universe, ['[500Weight]'], start_date, end_date)
    return merger(df_factor, df_index, how='left', to_replace={'500Weight': {np.nan: 0.0}})


def attach_risk_factors(universe, df_factor, start_date, end_date):
    df_risk = fetch_risk_factors(universe, RISK_FACTOR, start_date, end_date)
    return merger(df_factor, df_risk)


def attach_return_series(universe, df_factor, start_date, end_date):
    df_return = fetch_trading_info(
        universe, ['[Return]'], start_date, end_date)

    if not df_factor.empty:
        df = merger(df_factor, df_return, how='left',
                    to_replace={'Return': {np.nan: 0.0}})
    else:
        df = df_return

    df.rename(columns={'Return': 'dailyReturn'}, inplace=True)
    dates = sorted(df.index.unique())
    next_dates = pd.DataFrame({'date': dates, 'next': [*dates[1:], np.nan]})

    flat_table = df[['code', 'dailyReturn']].reset_index()
    flat_table = pd.merge(flat_table, next_dates, on='date')
    long_next_dates = flat_table[['code', 'next']]
    next_dates_return = pd.merge(long_next_dates, flat_table, how='left', left_on=[
        'code', 'next'], right_on=['code', 'date'])
    df['nextReturn1day'] = next_dates_return['dailyReturn'].values
    return df


def init_data_repository(start_date,
                         end_date,
                         factor_names,
                         incl_returns=True,
                         incl_risk_factors=True,
                         incl_index_component=True,
                         cal_chg=False):
    universe = set_universe(start_date, end_date)
    df_factor = fetch_factor_from_all_tables(
        universe, factor_names, start_date, end_date, cal_chg)

    if incl_index_component:
        df_factor = attach_index_components(
            universe, df_factor, start_date, end_date)

    if incl_risk_factors:
        df_factor = attach_risk_factors(
            universe, df_factor, start_date, end_date)

    if incl_returns:
        df_factor = attach_return_series(
            universe, df_factor, start_date, end_date)

    df = merger(df_factor, universe)
    df.dropna(inplace=True)
    return universe, df


if __name__ == "__main__":
    engine = create_conn(pm_config)
    sql = "select Date, Code from StockUniverse_500Enhance where [Date] >= '{start_date}' and [Date] <= '{end_date}'" \
        .format(start_date='20080101', end_date='20151231')

    df = pd.read_sql(sql, engine)

    sql = "select Date, Code from RiskFactors where [Date] >= '{start_date}' and [Date] <= '{end_date}'" \
        .format(start_date='20080101', end_date='20151231')

    df2 = pd.read_sql(sql, engine)

    pass

    # start_date = '2010-01-01'
    # end_date = '2015-12-31'
    #
    # import datetime as dt
    #
    # now = dt.datetime.today()
    # universe, repository = init_data_repository(start_date,
    #
    #                                             s
    # end_date,
    # [],  # MONEY_FACTOR,
    # # ['EODPrice'],# 'EMA05_returns', 'EMA005_returns'],
    # incl_returns = True,
    #                incl_risk_factors = False,
    #                                    incl_index_component = False,
    #                                                           cal_chg = False)
    # print(dt.datetime.today() - now)
    # print(repository)
