from setuptools import setup, find_packages

setup(name='piBreakDown',
      version='0.0.1',
      description='python version of iBreakDown',
      url='https://github.com/ModelOriented/piBreakDown',
      author='PW',
      author_email='dummy@mail',
      license='GPLv3',
      packages= find_packages(),
      install_requires=[
	'numpy==1.14.2',
	'pandas==0.23.4',
	'matplotlib==2.2.0'])