from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hypertab",
    author="Witold Wydmański",
    author_email="wwydmanski@gmail.com",
    version="0.2.2",
    description="HyperTab: hypernetwork for small tabular datasets",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT",
    packages=find_packages(include=['hypertab', 'hypertab.*']),
    install_requires=required,
    url="https://github.com/wwydmanski/hypertab"
)
