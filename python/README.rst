Kotekan Python Utilities
========================

Tools for interacting with kotekan via Python.

Python 3.12 is the minimum supported version, matching the Ubuntu 24.04 runtime
images. CI checks package installation and the debug viewer on Python 3.12.


Kotekan Runner and Tester
-------------------------

Just a quick note that for the `KotekanRunner` and `KotekanStageTester`
classes, if they are being run from a non-installed copy of the package, they
will attempt to use the installation of kotekan from within the build
directory of the same tree. If not, it will try to run a system installed
version.
