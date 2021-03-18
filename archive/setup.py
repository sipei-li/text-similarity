from setuptools import setup
from setuptools import find_packages

setup(name='textsimilarity',
      version='0.1',
      description='text similarity using graph matching',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      license='Privately owned/copyright',
      zip_safe=False)