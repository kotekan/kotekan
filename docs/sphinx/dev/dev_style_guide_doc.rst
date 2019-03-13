*************
Documentation
*************

All files should be liberally commented to allow others to debug,
and populated with full `doxygen <www.doxygen.org>`_ docstrings
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
   * @brief An FFTW-based F-engine stage.
   *  - fftwEngine : public kotekan::Stage
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

kotekan::Stages
+++++++++++++++
Should additionally describe their use of (and requirements for) config file options.
Special doxygen aliases exist to help make these explicit.

- ``@par Buffer`` should lead a section describing the buffers used by the stage

 - ``@buffer`` commands should list the input and output buffers or array of buffers

  - ``@buffer_format`` should describe the format of the buffer
  - ``@buffer_metadata`` should list the class of metadata used by the buffer

- ``@conf`` can be used for config file options, listing their types and default values.

These should be included following the detailed description, and before
any ``@warning``, ``@todo`` or ``@note`` strings.
A simple example comment follows:

.. code-block:: c++

  /**
   * @class exampleStage
   * @brief @c Kotekan Stage to demonstrate @c doxygen formatting.
   *
   * This is a detailed description of the example stage.
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
