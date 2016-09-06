# -*- coding: utf-8 -*-
u"""
Created on 2016-9-3

@author: cheng.li
"""

import time
import pyodbc
import aioodbc
import asyncio
import datetime as dt

loop = asyncio.get_event_loop()

dsn = 'DRIVER=FreeTDS;' \
      'SERVER=rm-bp1jv5xy8o62h2331o.sqlserver.rds.aliyuncs.com;' \
      'PORT=3433;' \
      'DATABASE=PortfolioManagements;' \
      'UID=wegamekinglc;' \
      'PWD=We051253524522;' \
      'TDS_Version=8.0;'
sql = 'select count(*) from [RiskFactor]'

async def fetch_async(sqls):
    conn = await aioodbc.connect(dsn=dsn, loop=loop)

    for sql in sqls:
        cur = await conn.cursor()
        await cur.execute(sql)
        r = await cur.fetchall()
        print(r)
        await cur.close()
    await conn.close()


def fetch_sync(sqls):
    conn = pyodbc.connect(dsn)
    for sql in sqls:
        cur = conn.cursor()
        cur.execute(sql)
        r = cur.fetchall()
        print(r)
        cur.close()
    conn.close()

iters = 50
sqls = [sql for _ in range(iters)]

start = dt.datetime.now()
loop.run_until_complete(fetch_async(sqls))
print('async elapsed: {0}'.format(dt.datetime.now() - start))

start = dt.datetime.now()

fetch_sync(sqls)

print('sync elapsed: {0}'.format(dt.datetime.now() - start))
