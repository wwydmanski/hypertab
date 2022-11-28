from setuptools import setup, find_packages

setup(
    name="tabular_hypernet",
    version="0.1.0",
    packages=find_packages(include=['tabular_hypernet', 'tabular_hypernet.*']),
    install_requires=["numpy >= 1.11.1", "matplotlib >= 1.5.1"],
)
