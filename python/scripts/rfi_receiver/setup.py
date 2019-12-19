#!/usr/bin/python3
"""
rfi_receiver, a server to receive real-time rfi data from kotekan.
``rfi_receiver`` lives on
`GitHub <https://github.com/kotekan/kotekan>`_.
"""

import setuptools
import versioneer


# Load the PEP508 formatted requirements from the requirements.txt file. Needs
# pip version > 19.0
with open("requirements.txt", "r") as fh:
    requires = fh.readlines()

# Now for the regular setuptools-y stuff
setuptools.setup(
    name="rfi_receiver",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="The CHIME Collaboration",
    author_email="willis@cita.utoronto.ca",
    description="A server to receive real-time RFI data",
    packages=["rfi_receiver"],
    license="GPL v3.0",
    url="http://github.com/kotekan/kotekan",
    install_requires=requires,
)
