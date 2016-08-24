# -*- coding: utf-8 -*-
u"""
Created on 2016-8-22

@author: cheng.li
"""

import pandas as pd


class InfoKeeper(object):

    def __init__(self) -> None:
        self.data_sets = []
        self.stored_data = pd.DataFrame()
        self.current_index = 0

    def attach_info(self, appended_data: pd.DataFrame) -> None:
        self.data_sets.append(appended_data)

    def info_view(self) -> pd.DataFrame:
        if self.current_index < len(self.data_sets):
            self.stored_data = self.stored_data.append(self.data_sets[self.current_index:])
            self.current_index = len(self.data_sets)
        return self.stored_data.copy(deep=True)
