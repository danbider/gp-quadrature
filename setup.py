"""
Setup script for the project.
"""

from setuptools import setup, find_packages

setup(
    name='gp-quadratures',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pydantic',
        'torch',
        'pytest',
    ],
    python_requires='>=3.10',
    description='A package for Gaussian Process quadratures',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Dan Biderman',
    author_email='danbider@gmail.com',
    url='https://github.com/danbider/gp-quadratures',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
    ],
)

