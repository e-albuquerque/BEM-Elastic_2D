from setuptools import setup, find_packages
setup(
    name='BEM-Elastic_2D',
    version='1.0',
    author='Eder Lima de Albuquerque',
    author_email='eder@unb.br',
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Optional
)