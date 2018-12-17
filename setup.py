try:
	from setuptools import setup
except:
	from distutils.core import setup

setup(
  name = 'findblas',
  packages = ['findblas'],
  version = '0.1.3',
  author = 'David Cortes',
  url = 'https://github.com/david-cortes/findblas',
  classifiers = [],
  data_files=[('include', ['findblas/findblas.h'])],
  include_package_data=True
) 
