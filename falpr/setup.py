from setuptools import setup

VERSION = '0.0.1'

with open('requirements.txt') as f:
    install_requires = f.readlines()

setup(
    name='falpr',
    version=VERSION,
    description='Fully Automated Licence Plate Recognizer',
    long_description='Magnificent app which recognizes chars in photo of licence plates',
    author='MB',
    url='https://github.com/matbur95/cypsio-pro',
    license='MIT',
    packages=['falpr'],
    install_requires=install_requires,
)
