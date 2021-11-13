import setuptools
import os
from dunamai import Version

class CleanCommand(setuptools.Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.egg-info')

git_version = Version.from_git().serialize(metadata=False)
VERSION = os.environ['VERSION'] if 'VERSION' in os.environ else git_version

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
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    cmdclass={'clean': CleanCommand}
)
