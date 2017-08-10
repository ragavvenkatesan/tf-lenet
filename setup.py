from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tf-lenet',
    version='0.1a1',
    description='Migration from theano to tensorflow',
    long_description=long_description,
    url='https://github.com/ragavvenkatesan/tf-lenet',
    author='Ragav Venkatesan',
    author_email='email@ragav.net',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Students, Researchers and Developers',
        'Topic :: Scientific/Engineering :: Computer Vision :: Deep Learning'
        'License :: MIT License',
        'Programming Language :: Python :: 2.7 :: Python 3.x',
    ],
    keywords='convolutional neural networks deep learning theano tensorflow',
    packages=find_packages(exclude=[]),
    install_requires=['tensorflow','numpy'],
    extras_require={
        'dev': ['sphinx', 'sphinx_rtd_theme', 'sphinxcontrib-bibtex'],
    },
    setup_requires=['tensorflow','numpy'],
    tests_require=['sphinx', 'sphinx_rtd_theme', 'sphinxcontrib-bibtex'],    
)