from setuptools import setup
import setuptools
import os
from Cython.Build import cythonize
import numpy as np

meta = {}
with open(os.path.join('petools', '__version__.py')) as f:
    exec(f.read(), meta)

setup(
    name=meta['__title__'],
    packages=setuptools.find_packages(),
    package_data={'petools': [
        'tools/estimate_tools/pafprocess/*.so',
        'tools/estimate_tools/pafprocess/*.pyd',
        'cpu/3d_converter_stats/*',
        'model_tools/transformers/human_processor/3d_converter_stats/*.npy',
        'model_tools/pose_classifier/process_data/classifier_stats/*.npy',
        'model_tools/human_tracker/similarity_based/body_proportions/data_statistics/*.npy'
    ]},
    version=meta['__version__'],
    description=meta['__description__'],
    long_description='...',
    author=meta['__author__'],
    author_email=meta['__contact__'],
    url=meta['__contact__'],
    include_package_data=True,  # This will include all files in MANIFEST.in in the package when installing.
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[],
    ext_modules=cythonize("**/*.pyx"),
    include_dirs=[np.get_include()]
)
