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
      'torch',
      'torch_ac',
      'NumPy',
      'gym',
      'pillow<7',
      'torchvision',
      'gym-unity==0.13.1',
      'mlagents==0.13.1',
      'mlagents-envs==0.13.1',
      'gym-minigrid',
   ],
)