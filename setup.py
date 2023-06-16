# type: ignore[no-typed-call]
"""Setup script for the flunet package."""

from setuptools import find_packages, setup

# Requirements definitions
SETUP_REQUIRES = [
    "setuptools>=66.0.0",
]

INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "pytorch-lightning>=2.0.0",
    "torchmetrics>=0.11.4",
    "hydra-core==1.3.2",
    "hydra-colorlog==1.2.0",
    "hydra-optuna-sweeper==1.2.0",
    "wandb>=0.15.3",
    "pyrootutils",
    "sh",
    "torch-scatter>=2.1.1",
    "torch-geometric>=2.3.1",
    "torch-cluster>=1.6.1",
    "scipy>=1.10.1",
]

INSTALL_DEV = [
    "black",
    "pytest>=7.3.1",
    "pytest-order",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.1",
    "mock>=5.0.2",
    "docformatter",
    "mypy",
    "mypy-parser",
    "pre-commit>=3.3.2",
    "rich",
    # "sphinx",
    # "sphinx-material",
    # "sphinx-autodoc-typehints",
    # "versioneer"
]

DEPENDENCY_LINKS = [
    "https://data.pyg.org/whl/torch-2.0.0+cu117.html",
]


# https://pypi.org/classifiers/
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.11",
    "Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.1" "License :: OSI Approved :: Apache Software License",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Operating System :: OS Independent",
]

setup(
    name="flunet",
    version="0.0.1",
    description=(
        "A common library for utilizing Graph Neural Networks (GNNs) in fluid dynamics research and applications."
    ),
    readme="README.md",
    license="Apache 2.0",
    author="The FluNet development team",
    url="https://github.com/Marimuda/FluNet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extra_require=INSTALL_DEV,
    dependency_links=DEPENDENCY_LINKS,
    classifiers=CLASSIFIERS,
)
