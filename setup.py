#!/usr/bin/env python
from setuptools import setup, find_packages

required = [
    'Box2D',
    'gym',
    'numpy'
]


setup(
    name='cliff-daredevil',
    version='0.0.0',
    packages=find_packages(),
    python_requires='>3.8',
    include_package_data=True,
    install_requires=required)
