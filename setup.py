from setuptools import setup

setup(
   name='app',
   version='1.0',
   description='',
   author='Iosif Nicolae',
   author_email='iosif@mailo.dev',
   packages=['app'],
   install_requires=[
      'torch',
      'torch_ac',
      'NumPy',
      'gym',
      'gym-minigrid',
   ],
)