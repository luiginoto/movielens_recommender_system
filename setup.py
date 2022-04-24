# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Group26_MoveLens',
    version='0.1.0',
    description='Package for first iteration of final project',
    long_description=readme,
    author='group26',
    author_email='ga947@nyu.edu',
    url='https://github.com/nyu-big-data/final-project-group-26',
    license=license,
    packages=find_packages(exclude=('tests', 'scratch', 'src'))
)