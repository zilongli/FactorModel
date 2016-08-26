# Factor Model

## 0. 依赖

  * *所有平台*

    * python 3.5

    * numpy

    * scipy

    * h5py

    * pandas

  * *Linux*

    * [``ipopt``](https://projects.coin-or.org/Ipopt)。安装目录需要是系统动态链接搜索目录。


  * *windows*

    ``lib`` 目录已经包含预编译好的``ipopt``包。

## 1. 安装

  * *Windows*

    * 将 ``lib`` 目录加入环境变量 ``PATH``;

    * 使用如下指令安装:

      ```
      python setup.py install
      ```


  * *Linux*

    * 将 ``lib``  目录加入环境变量 ``LD_LIBRARY_PATH``;

    * 使用如下指令安装:
      ```
      python setup.py install
      ```

## 2. Get Started
