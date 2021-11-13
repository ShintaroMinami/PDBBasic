import setuptools
from os import environ
from dunamai import Version

git_version = Version.from_git().serialize(metadata=False)
VERSION = environ['VERSION'] if 'VERSION' in environ else git_version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pdbbasic",
    version=VERSION,
    author="Shintaro Minami",
    author_email="shintaro.minami@gmail.com",
    description="Basic Utilities for Protein Structure Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShintaroMinami/PDBBasic",
    license='MIT',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'einops'
    ],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
    ],
)
