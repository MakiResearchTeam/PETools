from setuptools import setup
import setuptools

setup(
    name='PETools',
    packages=setuptools.find_packages(),
    package_data={'petools': [
        'tools/estimate_tools/pafprocess/*.so',
        'tools/estimate_tools/pafprocess/*.pyd',
        'gpu/3d_converter_stats/*',
        'cpu/3d_converter_stats/*',
        'model_tools/transformers/3d_converter_stats/*'
    ]},
    version='1.3.2',
    description='A set of tools to use pose estimation models',
    long_description='...',
    author='Kilbas Igor, Gribanov Danil',
    author_email='cool.danik01@yandex.ru',
    url='https://github.com/MakiResearchTeam/PETools.git',
    include_package_data=True,  # This will include all files in MANIFEST.in in the package when installing.
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ], install_requires=[]
)
