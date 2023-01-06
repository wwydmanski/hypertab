from setuptools import setup, find_packages

setup(
    name="hypertab",
    version="0.1.0",
    description="TBD",
    packages=find_packages(include=['hypertab', 'hypertab.*']),
    install_requires=["numpy >= 1.11.1", "matplotlib >= 1.5.1"],
)
