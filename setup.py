from setuptools import setup, find_packages

setup(
    name="slimdoc",
    version="1.0.0",
    packages=find_packages(),  # find_packages(include=['slimdoc', 'slimdoc.*'])
    url="https://github.com/marcel-lamott/SlimDoc/",
    python_requires=">=3.10",
)
