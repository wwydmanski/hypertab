from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="hypertab",
    version="0.1.0",
    description="TBD",
    packages=find_packages(include=['hypertab', 'hypertab.*']),
    install_requires=required,
)
