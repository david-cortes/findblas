try:
	from setuptools import setup
except:
	from distutils.core import setup

setup(
  name = 'findblas',
  packages = ['findblas'],
  version = '0.1.3.4',
  author = 'David Cortes',
  url = 'https://github.com/david-cortes/findblas',
  classifiers = [],
  data_files=[('include', ['findblas/findblas.h', 'findblas/rtd_mock.h'])],
  include_package_data=True
) 
