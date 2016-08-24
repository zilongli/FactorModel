import os
import sys
import glob
from setuptools import setup
from distutils import sysconfig
from pip.req import parse_requirements

requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')

install_reqs = parse_requirements(requirements_path, session=False)
reqs = [str(ir.req) for ir in install_reqs]

if os.name == "posix":
    exePath = sys.path
    for path in exePath:
        if path.endswith('site-packages'):
            packagePath = path
            break
else:
    packagePath = sysconfig.get_python_lib()

files = glob.glob("FactorModel/lib/*.*")
datafiles = [
    (os.path.join(packagePath, "FactorModel/lib"), files)]

setup(
    name='FactorModel',
    version='0.1.0',
    packages=['FactorModel'],
    url='https://coding.net/u/wegamekinglc/p/FactorModels',
    license='',
    author='cheng.li',
    author_email='wegamekinglc@hotmail.com',
    description='',
    install_requires=reqs,
    data_files=datafiles
)
