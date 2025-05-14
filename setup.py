from setuptools import setup, find_packages

setup(
    name="slimdoc",
    version="0.1.0",
    packages=find_packages(),  # find_packages(include=['slimdoc', 'slimdoc.*'])
    url="https://gitlab.cs.hs-rm.de/diss_lamott/slimdoc",
    python_requires=">=3.10",
)
