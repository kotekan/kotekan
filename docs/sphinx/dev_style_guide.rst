************
Style Guide
************

If you're going to work on **kotekan** code, please adhere to the following guidelines:


Files
------

New ``kotekanProcesses`` files should be placed in the ``lib/processes/`` folder,
and named in *CamelCase* to match the process class they contain.

Accelerator interfaces (such as e.g. OpenCL or CUDA) 
should create their own folder within ``lib/``,
for storage of kernel handling objects and the kernels themselves.

Support scripts and components can be placed in the ``script/`` folder.

Configuration (``yaml``) files should go in the ``config/`` folder.


General Structure
-----------------
Use 4-space tabs.

No trailing whitespace.



Documentation
-------------

All files should be liberally commented, with full `doxygen <www.doxygen.org>`_ docstrings
describing the class, variables, and all member functions.

Comment blocks should use an empty ``/**`` line to begin,
with subsequent lines beginning with ``_*_`` (space,star,space).
The comment block should close with a ``_*/`` (space,star,slash).

One-line docstrings (only to be used on private functions and variables)
should use the standard doxygen ``///`` (triple-slash).

Files
^^^^^^^^^^
Header files should begin with a quick summary block, identifying themselves
and listing their content functions or classes.

.. code-block:: c++

  /**
   * @file
   * @brief An FFTW-based F-engine process.
   *  - fftwEngine : public KotekanProcess
   */

Classes
^^^^^^^^^^
All classes should include a comment block immediately preceeding their declaration.

.. code-block:: c++

  /**
   * @class exampleClass
   * @brief Class with example @c doxygen formatting.
   *
   * A detailed description goes here.
   *
   * @todo    Make this actually do something.
   *
   * @author Keith Vanderlinde
   *
   */

``kotekanProcesses``
^^^^^^^^^^^^^^^^^^^^
Should additionally describe their use of (and requirements for) config file options.
Special doxygen aliases exist to help make these explicit.

- ``@par Buffer`` should lead a section describing the buffers used by the process

 - ``@buffer`` commands should list the input and output buffers or array of buffers

  - ``@buffer_format`` should describe the format of the buffer
  - ``@buffer_metadata`` should list the class of metadata used by the buffer

- ``@conf`` can be used for config file options, listing their types and default values.

These should be included following the detailed description, and before
any ``@warning``, ``@todo`` or ``@note`` strings.
A simple example comment follows:

.. code-block:: c++

  /**
   * @class exampleProcess
   * @brief @c kotekanProcess to demonstrate @c doxygen formatting.
   *
   * This is a detailed description of the example process.
   *
   * @par Buffers
   * @buffer in_buf Input kotekan buffer, to be consumed from.
   *     @buffer_format Array of @c datatype
   *     @buffer_metadata none, or a class derived from @c kotekanMetadata
   * @buffer out_buf Output kotekan buffer, to be produced into.
   *     @buffer_format Array of @c datatype
   *     @buffer_metadata none, or a class derived from @c kotekanMetadata
   *
   * @conf   config_param @c Int32 (Default: -7). A useful parameter
   *
   * @author Keith Vanderlinde
   *
   */



Naming
----------


Classes
^^^^^^^^^^
Classes in kotekan should use *CamelCase* formatting, e.g. ``myFavouriteProcess``.

Functions
^^^^^^^^^^
With the exception of class constructors & destructors, function names should use underscore notation,
e.g. ``my_func``.


Variables
^^^^^^^^^^
Variables in the code should use underscore naming, e.g. ``my_favourite_variable``.

Explicit typing should be used wherever possible, e.g. always use ``uint32_t`` rather than ``uint``.


Config File Settings
^^^^^^^^^^^^^^^^^^^^^^

Variables
+++++++++
Variables and parameters in the config file should use underscore naming, e.g. ``my_favourite_variable``.

If you need to reference the size of a datatype, use a C-like naming, e.g. ``sizeof_float = 4``.

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
