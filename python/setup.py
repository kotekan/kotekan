from setuptools import setup, find_packages

import kotekan

setup(
    name='kotekan',
    version=kotekan.__version__,
    license='MIT',

    packages=find_packages(),

    install_requires=['numpy>=1.7'],

   # entry_points="""
   #     [console_scripts]
   #     cora-makesky=cora.scripts.makesky:cli
   # """,
    author="Kotekan Developers",

    description="Python support code for kotekan",
    url="http://github.com/kotekan/kotekan/"
)
