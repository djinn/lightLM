#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from setuptools import setup, Extension
import io

__version__ = '0.1.0'

ext_modules = [
    Extension(
        "lightlm",
        sources=[
            "python/lightlm/lightlm.c",
            "c23/src/c_args.c",
        ],
        include_dirs=["c23/include"],
        language='c',
    ),
]

def _get_readme():
    """
    Use pandoc to generate rst from md.
    pandoc --from=markdown --to=rst --output=python/README.rst python/README.md
    """
    # For now, we'll just return the content of the top-level README
    with io.open("README.md", encoding='utf-8') as fid:
        return fid.read()


setup(
    name='lightlm',
    version=__version__,
    author='Jules',
    author_email='',
    description='lightlm Python bindings',
    long_description=_get_readme(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    url='https://github.com/facebookresearch/fastText',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    packages=['lightlm'],
    package_dir={'': 'python'},
    zip_safe=False,
)
