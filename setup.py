from setuptools import setup

setup(
   name='app',
   version='1.0',
   description='',
   author='Iosif Nicolae',
   author_email='iosif@mailo.dev',
   packages=['app'],
   install_requires=[
      'getkey',
      'pyyaml',
      'torch==1.4.0',
      'NumPy',
      'gym',
      'pillow<7',
      'torchvision',
      'gym-unity',
      'mlagents==0.13.1',
      'mlagents-envs==0.13.1',
      'gym-minigrid',
   ],
)