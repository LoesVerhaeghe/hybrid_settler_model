from setuptools import setup, find_packages

setup(
    name='parallel_settler_model',
    version='0.1',
    packages=find_packages(include=['src', 'src.*', 'utils', 'utils.*', 'scripts']),
)