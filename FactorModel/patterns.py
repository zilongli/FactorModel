u"""
Created on 2016-9-12

@author: cheng.li
"""


class Singleton(type):

    instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.instances:
            cls.instances[cls] = \
                super(Singleton, cls).__call__(*args, *kwargs)
        return cls.instances[cls]
