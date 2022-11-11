from setuptools import find_packages, setup

setup(
    name='politician2vec',
    version='0.0.1',
    author='Mathias Bruun',
    author_email='pvf607@alumni.ku.dk',
    packages=['politician2vec'],
    scripts=[],
    url='https://github.com/mathiasbruun/politician2vec',
    license='docs/LICENSE.txt',
    description='Distributed representations of words, docs, and politicians',
    long_description=open('README.md').read(),
    install_requires=[
#        'numpy==1.20.3',
#        'numba==0.53.0',
    ],
)