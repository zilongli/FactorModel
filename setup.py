import os
from setuptools import setup
from pip.req import parse_requirements

requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')

install_reqs = parse_requirements(requirements_path, session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='FactorModel',
    version='0.1.0',
    packages=['FactorModel'],
    url='https://coding.net/u/wegamekinglc/p/FactorModels',
    license='',
    author='cheng.li',
    author_email='wegamekinglc@hotmail.com',
    description='',
    install_requires=reqs
)
