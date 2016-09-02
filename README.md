# Factor Model

## 依赖

### 所有平台

* h5py
* matplotlib
* numpy
* pandas
* python 3.5
* scipy

### Linux

* [``ipopt``](https://projects.coin-or.org/Ipopt)。安装目录需要是系统动态链接搜索目录。

### windows

``lib`` 目录已经包含预编译好的``ipopt``包。


## 安装

### Windows

* 将 ``lib`` 目录加入环境变量 ``PATH``;

* 使用如下指令安装:

```shell
python setup.py install
```


### Linux

* 将 ``lib``  目录加入环境变量 ``LD_LIBRARY_PATH``;

* 使用如下指令安装:

```shell
python setup.py install
```


## Get Started

可以直接运行的例子可以``FactorModel/examples``下找到，例如运行：

```shell
python FactorModel/examples/examples_101.py
```
