#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Describe Your Project",
    author="jákup Svøðstein",
    author_email="jakupsv@setur.fo",
    url="https://github.com/Marimuda/FluNet",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
