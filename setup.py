#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup script for vectorface."""

__author__ = 'Aaron Hosford'

from setuptools import setup
from codecs import open
from os import path

from vectorface import __version__

here = path.abspath(path.dirname(__file__))


# Default long description
long_description = """

Vectorface
==========

*Vector computing with arbitrary backends*

Vectorface is a collection of standardized interfaces and adapters for vector 
computing libraries such as numpy, tensorflow, pytorch, and cupy. It is 
intended to make swapping out vector computing libraries completely transparent
to client code, and also to facilitate interoperability between them.

**Currently, only 1D vectors are supported.** This is a major limitation on the
utility of this package. Support for multidimensional arrays (matrices and 
tensors) is a top priority and will be added in a future release.


Links
-----

-  `Source <https://github.com/hosford42/vectorface>`__
-  `Distribution <https://pypi.python.org/pypi/vectorface>`__

The package is available for download under the permissive `Revised BSD
License <https://github.com/hosford42/vectorface/blob/master/LICENSE>`__.

""".strip()


# Get the long description from the relevant file. First try README.rst,
# then fall back on the default string defined here in this file.
if path.isfile(path.join(here, 'README.rst')):
    with open(path.join(here, 'README.rst'),
              encoding='utf-8',
              mode='r') as description_file:
        long_description = description_file.read()

# See https://pythonhosted.org/setuptools/setuptools.html for a full list
# of parameters and their meanings.
setup(
    name='vectorface',
    version=__version__,
    author=__author__,
    author_email='hosford42@gmail.com',
    url='http://hosford42.github.io/vectorface',
    license='Revised BSD',
    platforms=['any'],
    description='Vectorface: Vector computing with arbitrary backends',
    long_description=long_description,

    # See https://pypi.python.org/pypi?:action=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Environment :: GPU',
        'Environment :: Plugins',
        'Operating System :: OS Independent',
        'Topic :: Adaptive Technologies',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Object Brokering',
        'Topic :: Utilities',
        'Typing :: Typed',

        # Specify the Python versions you support here. In particular,
        # ensure that you indicate whether you support Python 2, Python 3
        # or both.
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.12',
    ],

    keywords='vector array numpy backend compute engine tensorflow cuda torch pytorch cupy',
    packages=['vectorface'],

    test_suite="test_vectorface",
    tests_require=["numpy"],
)
