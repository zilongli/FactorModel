# -*- coding: utf-8 -*-
u"""
Created on 2016-8-24

@author: cheng.li
"""

import sys
import unittest
from FactorModel.tests.test_optimizer import TestOptimizer


def test():
    print('Python ' + sys.version)
    suite = unittest.TestSuite()

    tests = unittest.TestLoader().loadTestsFromTestCase(TestOptimizer)
    suite.addTests(tests)

    res = unittest.TextTestRunner(verbosity=3).run(suite)
    if len(res.errors) >= 1 or len(res.failures) >= 1:
        sys.exit(-1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    test()
