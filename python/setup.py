from os import path
from setuptools import setup, find_packages
import versioneer

here = path.abspath(path.dirname(__file__))


# Get the long description from the README file
with open(path.join(here, "README.rst"), "r") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), "r") as f:
    requirements = f.readlines()

setup(
    name="kotekan",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    author="Kotekan Developers",
    description="Python support code for kotekan",
    long_description=long_description,
    url="http://github.com/kotekan/kotekan/",
    packages=find_packages(),
    py_modules=["_version"],
    install_requires=requirements,
    entry_points="""
        [console_scripts]
        kotekan-ctl=kotekan.scripts.ctl:cli
        polyco-tools=kotekan.scripts.polyco_tools:cli
        baseband-archiver=kotekan.scripts.baseband_archiver:cli
    """,
    scripts=["scripts/rfi_receiver/rfi_receiver.py"],
)
