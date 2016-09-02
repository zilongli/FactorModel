# -*- coding: utf-8 -*-
u"""
Created on 2016-8-16

@author: cheng.li
"""

import abc
import copy
from typing import List
from typing import Optional
from typing import Tuple
import datetime as dt
import pandas as pd
import sqlalchemy
from FactorModel.utilities import load_mat
from FactorModel.utilities import format_date_index
from FactorModel.utilities import format_date_to_index

DB_DRIVER_TYPE = 'pyodbc'

ALPHA_FACTOR_TABLES = {'AlphaFactors_PROD',
                       'AlphaFactors_Difeiyue',
                       'AlphaFactors_Licheng',
                       'AlphaFactors_Lishun',
                       'AlphaFactors_101'}


class Provider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __iter__(self):
        pass


class DataFrameProvider(Provider):
    def __init__(self) -> None:
        self.calc_date_list = None
        self.apply_date_list = None
        self.repository = None
        self.corr_mat = pd.DataFrame()
        self.factor_vol = pd.DataFrame()
        self.risk_level = pd.DataFrame()
        self.risk_style = pd.DataFrame()
        self.date_table = pd.DataFrame()

    @property
    def source_data(self) -> pd.DataFrame:
        return self.repository.copy(deep=True)

    def calc_dates(self) -> List[pd.Timestamp]:
        return copy.deepcopy(self.calc_date_list)

    def apply_dates(self) -> List[pd.Timestamp]:
        return copy.deepcopy(self.apply_date_list)

    def __iter__(self):
        for calc_date, apply_date in zip(self.calc_date_list, self.apply_date_list):
            yield calc_date, apply_date, self.fetch_values_from_repo(apply_date)

    def fetch_values_from_repo(self, date: pd.Timestamp,
                               date_type: Optional[str] = 'apply_date',
                               fields: Optional[List[str]] = None) -> pd.DataFrame:
        if date_type.lower() == 'apply_date':
            if fields:
                return self.repository.loc[date, fields]
            else:
                return self.repository.loc[date, :]

        elif date_type.lower() == 'calc_date':
            flags = self.repository.calcDate == date
        else:
            raise ValueError("{date_type} is not recognized."
                             .format(date_type=date_type))

        if fields:
            return self.repository.loc[flags, fields]
        else:
            return self.repository.loc[flags, :]

    def fetch_factor_corr(self, calc_date):
        return self.corr_mat.loc[self.corr_mat.index.asof(calc_date)].values[0]

    def fetch_factor_vol(self, calc_date):
        return self.factor_vol.loc[self.factor_vol.index.asof(calc_date)].values[0]

    def fetch_risk_level(self, calc_date):
        return self.risk_level.loc[self.risk_level.index.asof(calc_date)].values

    def fetch_risk_style(self, calc_date, fields):
        return self.risk_style.loc[self.risk_style.index.asof(calc_date), fields].values


class FileProvider(DataFrameProvider):
    def __init__(self, file_path: str, rows: Optional[int] = None):
        super().__init__()
        self.repository = load_mat(file_path, rows)
        self.calc_date_list = self.repository.calcDate.unique()
        self.apply_date_list = self.repository.applyDate.unique()


class DBProvider(DataFrameProvider):
    def __init__(self,
                 server,
                 user,
                 pwd):
        super().__init__()
        if DB_DRIVER_TYPE == 'pymssql':
            conn_template = 'mssql+pymssql://{user}:{pwd}@{server}/{db_name}'
        elif DB_DRIVER_TYPE == 'pyodbc':
            conn_template = 'mssql+pyodbc://{user}:{pwd}@{server}/{db_name}?driver=SQL+Server+Native+Client+11.0'
        mf_conn = conn_template.format(user=user, pwd=pwd, server=server, db_name='MultiFactor')
        pm_conn = conn_template.format(user=user, pwd=pwd, server=server, db_name='PortfolioManagements')

        self.mf_engine = sqlalchemy.create_engine(mf_conn)
        self.pm_engine = sqlalchemy.create_engine(pm_conn)

    def load_date_table(self,
                        start_date: int,
                        end_date: int) -> None:
        sql = 'select [Date] as [date], EOM as eom, EOW as eow from Timeline ' \
              'where [Date] >= {start_date} and [Date] <= {end_date}'.format(start_date=start_date, end_date=end_date)
        self.date_table = pd.read_sql(sql, self.mf_engine)
        format_date_to_index(self.date_table, 'date', as_index=True)

    def load_repository_data(self,
                              start_date: int,
                              end_date: int,
                              alpha_factors: List[str]) -> Tuple[int, int]:
        # stock universe
        sql = 'select calcDate, applyDate, Code as code from [StockUniverse] ' \
              'where [applyDate] >= {apply_start} and [applyDate] <= {apply_end} ' \
              'ORDER BY [applyDate], [code]' \
            .format(apply_start=start_date, apply_end=end_date)

        stock_universe = pd.read_sql(sql, self.pm_engine)

        calc_dates = stock_universe.calcDate.unique()
        calc_start = calc_dates[0]
        calc_end = calc_dates[-1]
        code_list = stock_universe.code.unique()
        code_list_str = ','.join(map(lambda x: str(x), code_list))

        # risk and factor
        sql = 'select * from [RiskFactor] ' \
              'where [date] >= {calc_start} and [date] <= {calc_end} and [code] in ({code_list_str}) ' \
              'ORDER BY [date], [code]' \
            .format(calc_start=calc_start, calc_end=calc_end, code_list_str=code_list_str)
        raw_risk_data = pd.read_sql(sql, self.pm_engine)
        df = pd.merge(stock_universe, raw_risk_data, how='left', left_on=['calcDate', 'code'],
                      right_on=['Date', 'Code'], suffixes=('', '_y'))
        df.drop(['Date', 'Code'], axis=1, inplace=True)

        sql_template = "select [name] from [syscolumns] where [id] = object_id('{alpha_factor_table}')"
        raw_factor_dict = {
            table: pd.read_sql(sql_template.format(alpha_factor_table=table), self.pm_engine)['name'].values[2:] for table in
            ALPHA_FACTOR_TABLES}
        factor_dict = dict()
        for key, values in raw_factor_dict.items():
            for v in values:
                factor_dict[v] = key

        input_factor_map = {factor: factor_dict[factor] for factor in alpha_factors}
        table2factors = dict()
        for factor, table_name in input_factor_map.items():
            if table_name in table2factors:
                table2factors[table_name].append(factor)
            else:
                table2factors[table_name] = [factor]

        for table_name, factors in table2factors.items():
            factor_str = ','.join(factors)
            sql = 'select [Date], [Code], {factors} from {table_name} ' \
                  'where [Date] >= {calc_start} and [Date] <= {calc_end} and [Code] in ({code_list_str}) ' \
                  'ORDER BY [Date], [Code]' \
                .format(factors=factor_str, table_name=table_name, calc_start=calc_start, calc_end=calc_end,
                        code_list_str=code_list_str)
            factor_data = pd.read_sql(sql, self.pm_engine)
            df = pd.merge(df, factor_data, how='left', left_on=['calcDate', 'code'], right_on=['Date', 'Code'],
                          suffixes=('', '_y'))
            df.drop(['Date', 'Code'], axis=1, inplace=True)

        # future returns and res
        sql = 'select [Date], [Code], [D1Res], [D5Res], [D10Res], [D15Res], [D20Res] from [StockResidual] ' \
              'where [Date] >= {calc_start} and [Date] <= {calc_end} and [Code] in ({code_list_str}) ' \
              'ORDER BY [Date], [Code]' \
            .format(calc_start=calc_start, calc_end=calc_end, code_list_str=code_list_str)
        raw_future_res = pd.read_sql(sql, self.pm_engine)
        df = pd.merge(df, raw_future_res, how='left', left_on=['calcDate', 'code'], right_on=['Date', 'Code'],
                      suffixes=('', '_y'))
        df.drop(['Date', 'Code'], axis=1, inplace=True)

        sql = 'select [Date], [Code], [D1LogReturn], [D5LogReturn], [D10LogReturn], [D15LogReturn], [D20LogReturn] ' \
              'from [StockReturns] ' \
              'where [Date] >= {calc_start} and [Date] <= {calc_end} and [Code] in ({code_list_str}) ' \
              'ORDER BY [Date], [Code]' \
            .format(calc_start=calc_start, calc_end=calc_end, code_list_str=code_list_str)
        raw_future_returns = pd.read_sql(sql, self.pm_engine)
        df = pd.merge(df, raw_future_returns, how='left', left_on=['calcDate', 'code'], right_on=['Date', 'Code'],
                      suffixes=('', '_y'))
        df.drop(['Date', 'Code'], axis=1, inplace=True)

        # daily return
        start_date_dt = dt.datetime.strptime(str(start_date), '%Y%m%d')
        end_date_dt = dt.datetime.strptime(str(end_date), '%Y%m%d')

        offset_start_date = start_date_dt - dt.timedelta(days=30)
        offset_end_date = end_date_dt + dt.timedelta(days=30)

        sql = 'select [Date], [Code], [Return] as dailyReturn from [TradingInfo1] ' \
              'where [Date] >= {start_date} and [Date] <= {end_date} and [Code] in ({code_list_str}) ' \
              'order by [Date], [Code]' \
            .format(start_date=offset_start_date.strftime('%Y%m%d'), end_date=offset_end_date.strftime('%Y%m%d'),
                    code_list_str=code_list_str)
        raw_returns = pd.read_sql(sql, self.mf_engine)

        dates_list = raw_returns.Date.unique()
        next_dates_list = [*dates_list[1:], None]
        dates_table = pd.DataFrame(data={'this': dates_list, 'next': next_dates_list}, dtype=int)

        dates_code_matched = pd.merge(raw_returns[['Date', 'Code']], dates_table, how='left', left_on=['Date'],
                                      right_on=['this'])
        next_returns = pd.merge(dates_code_matched, raw_returns[['Date', 'Code', 'dailyReturn']], how='left',
                                left_on=['next', 'Code'], right_on=['Date', 'Code'])
        raw_returns['nextReturn1day'] = next_returns['dailyReturn']
        df = pd.merge(df, raw_returns, how='left', left_on=['applyDate', 'code'], right_on=['Date', 'Code'],
                      suffixes=('', '_y'))
        df.drop(['Date', 'Code'], axis=1, inplace=True)

        # index components
        sql = 'select [Date], [Code], [500Weight] as zz500 from [IndexComponents] ' \
              'where [Date] >= {calc_start} and [Date] <= {calc_end} and [Code] in ({code_list_str}) ' \
              'order by [Date], [Code]' \
            .format(calc_start=calc_start, calc_end=calc_end, code_list_str=code_list_str)
        raw_index_components = pd.read_sql(sql, self.mf_engine)
        raw_index_components['zz500'] /= 100.
        df = pd.merge(df, raw_index_components, how='left', left_on=['calcDate', 'code'], right_on=['Date', 'Code'],
                      suffixes=('', '_y'))
        df.drop(['Date', 'Code'], axis=1, inplace=True)

        # market risk
        df['market'] = 1.

        # suspend info
        sql = 'select [Date], [Code], [Suspend20DayTrailing], [Suspend5DayTrailing] from TradingFlagFactor ' \
              'where [Date] >= {calc_start} and Date <= {calc_end} and [Code] in ({code_list_str}) ' \
              'order by [Date], [Code]' \
            .format(calc_start=calc_start, calc_end=calc_end, code_list_str=code_list_str)
        raw_suspend_flags = pd.read_sql(sql, self.mf_engine)
        df = pd.merge(df, raw_suspend_flags, how='left', left_on=['calcDate', 'code'], right_on=['Date', 'Code'],
                      suffixes=('', '_y'))
        df.drop(['Date', 'Code'], axis=1, inplace=True)
        df.fillna(0, inplace=True)

        # format data
        format_date_to_index(df, 'applyDate')
        format_date_to_index(df, 'calcDate')
        df['code'] = df['code'].astype(int)
        df.set_index(['applyDate'], drop=False, inplace=True)

        self.calc_date_list = df.calcDate.unique()
        self.apply_date_list = df.applyDate.unique()
        self.repository = df

        return calc_start, calc_end

    def load_cov_data(self,
                       calc_start: int,
                       calc_end: int) -> None:
        # correlation matrix
        sql = 'select * from  [CorrelationMatrix] where [Date] >= {calc_start} and [Date] <= {calc_end} order by [Date]' \
            .format(calc_start=calc_start, calc_end=calc_end)
        raw_corr_data = pd.read_sql(sql, self.pm_engine).values
        model_dates = raw_corr_data[:, 0]
        all_corr_matrixs = []
        for i, _ in enumerate(model_dates):
            matrix = raw_corr_data[i, 1:]
            matrix.shape = 30, 30
            all_corr_matrixs.append(matrix)
        self.corr_mat = pd.DataFrame(data={'date': model_dates, 'matrix': all_corr_matrixs})
        self.corr_mat.set_index('date', inplace=True)
        format_date_index(self.corr_mat)

        # factor vol model
        sql = 'select * from [FactorVolatility] where [Date] >= {calc_start} and [Date] <= {calc_end} order by [Date]' \
            .format(calc_start=calc_start, calc_end=calc_end)
        raw_factor_vol = pd.read_sql(sql, self.pm_engine).values
        model_dates = raw_factor_vol[:, 0]
        all_factor_vol = []
        for i, _ in enumerate(model_dates):
            all_factor_vol.append(raw_factor_vol[i, 1:])
        self.factor_vol = pd.DataFrame({'date': model_dates, 'vol': all_factor_vol})
        self.factor_vol.set_index('date', inplace=True)
        format_date_index(self.factor_vol)

        # special risk level model
        sql = 'select [Date] as [date], level from [SpecialRiskLevel] where [Date] >= {calc_start} and [Date] <= {calc_end} order by [Date]' \
            .format(calc_start=calc_start, calc_end=calc_end)
        self.risk_level = pd.read_sql(sql, self.pm_engine, index_col=['date'])
        format_date_index(self.risk_level)

        # special risk style model
        sql = 'select * from [SpecialRiskStyleModel] where [Date] >= {calc_start} and [Date] <= {calc_end} order by [Date]' \
            .format(calc_start=calc_start, calc_end=calc_end)
        raw_risk_style = pd.read_sql(sql, self.pm_engine, index_col='Date')
        self.risk_style = raw_risk_style
        format_date_index(self.risk_style)

    def load_data(self,
                   start_date: str,
                   end_date: str,
                   alpha_factors: List[str]) :
        start_date = int(start_date.replace('-', ''))
        end_date = int(end_date.replace('-', ''))
        calc_start, calc_end = self.load_repository_data(start_date, end_date, alpha_factors)
        self.load_cov_data(calc_start, calc_end)
        self.load_date_table(start_date, end_date)


if __name__ == "__main__":
    db_provider = DBProvider('10.63.6.219', 'sa', 'A12345678!')
    db_provider.load_data('2008-01-02', '2010-11-01', ['Growth', 'CFinc1', 'Rev5m'])
    print(db_provider.risk_style.index[0])

    calc_date = pd.Timestamp('2010-01-30')
    print(db_provider.fetch_factor_corr(calc_date))
