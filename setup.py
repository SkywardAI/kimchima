from setuptools import setup, find_packages

setup(
    name='kimchima',
    version='0.2.0',
    author='Aisuko',
    author_email='urakiny@gmail.com',
    description='A collection of tools for testing and dump llama2',
    packages=find_packages(),
    install_requires=[
        'torch',
        'sentencepiece',
        'tqdm',
    ],
)