from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='photobox',
    packages=['photobox'],
    description='Photobox Software - MeBioS',
    version='1.0',
    url='https://github.com/kalfasyan/photobox.git',
    author='Yannis Kalfas',
    author_email='kalfasyan@gmail.com',
    keywords=['sticky-plates','insect-plates','insects','photobox','mebios']
    packages=['photobox'],
    install_requires=['requests'],
    )