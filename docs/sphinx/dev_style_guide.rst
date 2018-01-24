************
Style Guide
************

If you're going to work on **kotekan** code, please adhere to the following guidelines:

Files
------

Names & locations.



Documentation
-------------

All files should be liberally commented, with full `doxygen <www.doxygen.org>`_ docstrings
describing the class, variables, and all member functions.

``kotekanProcesses``
^^^^^^^^^^^^^^^^^^^^
Should additionally describe their use of (and requirements for) config file options,
using ``@conf`` entries.

.. todo::
   INSERT EXAMPLE HERE


Naming
----------


Classes
^^^^^^^^^^
Classes in kotekan should use *CamelCase* formatting, e.g. ``myFavouriteProcess``.


Config File Settings
^^^^^^^^^^^^^^^^^^^^^^

Variables
+++++++++
Variables and parameters in the config file should use underscore naming, e.g. ``my_favourite_variable``.

``Buffers``
+++++++++++
``Buffers`` used by a ``KotekanProcess`` should be named ``in_buf`` or ``out_buf``,
for input (being produced into) and output (being consumed from), respectively.

When multiple ``Buffers`` of a similar type are needed,
they should go into an array of buffers, similarly named.

If multiple ``Buffers`` of different types are used by a single process,
a short identifier should be appended to their names,
e.g. ``in_buf_voltages`` and ``in_buf_pointing``.


Variables
^^^^^^^^^^

Structs
^^^^^^^^^^

Enums
^^^^^^^^^^
