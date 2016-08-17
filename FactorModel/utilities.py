# -*- coding: utf-8 -*-
u"""
Created on 2016-5-6

@author: cheng.li
"""

import logging
import functools
import pandas as pd
import numpy as np
import h5py


def load_mat(file_path):
    f = h5py.File(file_path)
    col_data = f.get('cols')[:][0]
    def read_cols(col_data):
        headers = []
        for ref in f.get('cols')[:][:]:
            obj = f[ref[0]]
            head = ''.join(chr(i[0]) for i in obj[:])
            headers.append(head)
        return headers

    headers = read_cols(col_data)
    df = pd.DataFrame(f.get('data')[:].T,
    columns=headers)
    return df


def py_assert(cond, exep_type, msg):
    if not cond:
        raise exep_type(msg)


def list_to_str(names):
    return ','.join(names)


def format_date_to_index(df, col_name, formater='%Y%m%d'):
    df[col_name] = pd.to_datetime(df[col_name], format=formater)
    df.set_index(col_name, drop=True, inplace=True)
    return df


def format_codes(df):
    df.code = df.code.apply(lambda x: "{0:06d}".format(x))
    return df


def merger(left, right, how='inner', to_replace=None):
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
def create_logger():
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

logger = create_logger()


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

if __name__ == "__main__":
    import datetime as dt

    df = load_mat('d:/data.mat')
    calcDates = df.calcDate.unique()

    start = dt.datetime.now()
    for calcDate in calcDates:
        df.loc[df.calcDate == calcDate, :]
    print(dt.datetime.now() - start)
