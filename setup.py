#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='ActiveNoise',
    version='0.1.0',
    packages=find_packages(include=['ActiveNoise', 'ActiveNoise.*'])
)
