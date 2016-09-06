# -*- coding: utf-8 -*-
u"""
Created on 2016-5-6

@author: cheng.li
"""

from typing import Optional
from typing import List
from typing import Any
import logging
import pandas as pd
import numpy as np


def combine(df1: pd.DataFrame,
            df2: pd.DataFrame,
            inplace=False) -> pd.DataFrame:
    if inplace:
        df1.set_index('code', append=True, drop=False, inplace=inplace)
        df2.set_index('code', append=True, inplace=inplace)
    else:
        df1 = df1.set_index('code', append=True, drop=False, inplace=inplace)
        df2 = df2.set_index('code', append=True, inplace=inplace)
    df1[df2.columns] = df2
    df1.dropna(inplace=True)
    df1.reset_index(level=-1, drop=True, inplace=True)
    return df1


def py_assert(cond: bool, exep_type: Any, msg: str) -> None:
    if not cond:
        raise exep_type(msg)


def list_to_str(names: List[str]) -> str:
    return ','.join(names)


def format_date_index(df: pd.DataFrame,
                      formater: Optional[str]='%Y%m%d') -> None:
    df.index = pd.to_datetime(df.index, format=formater)


def format_date_to_index(df: pd.DataFrame,
                         col_name: str,
                         formater: Optional[str]='%Y%m%d',
                         as_index: Optional[bool]=False) -> None:
    df[col_name] = pd.to_datetime(df[col_name], format=formater)
    if as_index:
        df.set_index(col_name, drop=False, inplace=True)


def format_codes(df: pd.DataFrame) -> None:
    df.code = df.code.apply(lambda x: "{0:06d}".format(x))


def merger(left: pd.DataFrame,
           right: pd.DataFrame,
           how: Optional[str]='inner',
           to_replace: Optional[dict]=None) -> pd.DataFrame:
    left_on = [np.array(left.index), left.code]
    right_on = [np.array(right.index), right.code]

    df = pd.merge(left, right, how=how, left_on=left_on,
                  right_on=right_on).drop('code_y', axis=1)
    df.rename(columns={'key_0': 'date', 'code_x': 'code'}, inplace=True)
    df.set_index('date', inplace=True)

    if to_replace:
        df.replace(to_replace, inplace=True)
    return df


# exception catching stuff
def create_logger() -> logging.Logger:
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("exception_logger")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    fh = logging.FileHandler(r"test.log")

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)
    return logger


def exception(logger):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur

    @param logger: The logging object
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                # log the exception
                err = "There was an exception in  "
                err += func.__name__
                logger.exception(err)

            # re-raise the exception
            raise
        return wrapper
    return decorator
