# -*- coding: utf-8 -*-
u"""
Created on 2016-5-26

@author: cheng.li
"""

import pandas as pd
import numpy as np
from FactorModel.utilities import pm_config
from FactorModel.utilities import create_conn
from FactorModel.loader import init_data_repository
from FactorModel.facts import TA_FACTOR


def filter_factors(factor_list):
    factor_list = list(factor_list)
    res = []
    for fac in factor_list:
        if not fac.endswith('_diff'):
            res.append(fac)
    return res


def naming_suffix(df, suffix='_diff'):
    naming_dict = {}
    for col in df.columns:
        if col != 'code':
            naming_dict[col] = col + suffix
    df.rename(columns=naming_dict, inplace=True)
    return df


if __name__ == "__main__":
    start_date = '2008-01-01'
    end_date = '2015-12-31'

    TA_FACTOR = list(TA_FACTOR)
    factor_list = filter_factors(TA_FACTOR)

    universe, repository = init_data_repository(start_date,
                                                end_date,
                                                factor_list,
                                                True,
                                                False,
                                                False,
                                                True)

    repository = naming_suffix(repository)

    engine = create_conn(pm_config)
    cursor = engine.cursor()

    col_insert_sql_template = \
        """ALTER TABLE PortfolioManagements.dbo.AlphaFactors_Licheng ADD {fac_name} FLOAT null;"""

    for fac in repository.columns[1:]:
        if not fac.endswith('_diff_diff'):
            if fac not in TA_FACTOR:
                sql = col_insert_sql_template.format(fac_name=fac)
                print(sql)
                cursor.execute(sql)

    target_cols = []
    for fac in repository.columns[1:]:
        if fac.endswith('_diff') and not fac.endswith('_diff_diff'):
            target_cols.append(fac)

    row_update_sql_template = \
        "UPDATE PortfolioManagements.dbo.AlphaFactors_Licheng set"

    for fac in target_cols:
        row_update_sql_template += " " + fac + "=%(" + fac + ")s,"

    row_update_sql_template = row_update_sql_template[:-1] \
                              + " where Date=%(date)s and" \
                              + " Code=%(code)s"

    date_ints = list(map(lambda x: int(x.strftime("%Y%m%d")), repository.index))
    code_int = list(repository.code.astype(int))
    factor_values = repository[target_cols].values
    all_values = []
    for i, date in enumerate(date_ints):
        code = int(code_int[i])
        values = list(factor_values[i, :])
        values = {c: v for c, v in zip(target_cols, values)}
        values['date'] = date
        values['code'] = code
        all_values.append(values)

    del repository
    print(len(all_values))
    cursor.executemany(row_update_sql_template, all_values)
    engine.commit()
    engine.close()
    #print(repository)
