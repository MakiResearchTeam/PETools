from setuptools import setup
import setuptools

setup(
    name='PETools',
    packages=setuptools.find_packages(),
    package_data={'petools': ['tools/estimate_tools/pafprocess/*.so', 'gpu/3d_converter_stats/*']},
    version='0.5.0',
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