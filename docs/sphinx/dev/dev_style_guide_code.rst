C++ Code Guidelines
---------------------

For everything design related, you should follow the `C++ Core Guidelines
<http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines>`_. Additionally
to that, the following rules mostly describe the code formatting used in this
project.

Naming
^^^^^^^^^^
- Type names in kotekan should be nouns, use *CamelCase* formatting and begin
  with a capital letter, e.g. ``MyFavouriteStage``. This includes classes,
  enums, structs and typedefs.
- With the exception of class constructors & destructors, function names should
  use underscore notation and begin with a lower case letter, e.g. ``my_func``.

Variables
^^^^^^^^^^
- Variables in the code should use underscore naming, e.g.
  ``my_favourite_variable``.

- Explicit typing should be used wherever possible, e.g. always use ``uint32_t``
  rather than ``uint``.

- Variables that derive from config values start with an underscore, e.g.
  ``_my_config_variable``.

Namespaces
^^^^^^^^^^
- Never do ``using namespace X;`` in header files

- Also never do ``using namespace std;``.

- Use ``std::begin`` and ``std::end`` instead of ``.begin()`` and ``.end()``.

Header files
^^^^^^^^^^^^^
C++ header files should be named after the class they describe and use the
ending `.hpp`. They should begin with a header file commit like

.. code-block:: c++

    /****************************************************
    @file
    @brief Brief description of the classes in this file.
    - MyClass
    - MyOtherClass : public MyOtherBaseClass
    *****************************************************/

Immediately after the header file comment include guards should follow:

.. code-block:: c++

    #ifndef MY_CLASS_H
    #define MY_CLASS_H

With a simple

.. code-block:: c++

    #endif

at the end of the file.


#include Style
^^^^^^^^^^^^^^^
The list of #includes should be in the order:

 - Header file implementing the interface of this `.cpp` file (if applicable).
 - Local/Private Headers
 - kotekan project headers
 - System #includes

Each category should be sorted lexicographically by the full path.

The small details
^^^^^^^^^^^^^^^^^^

- Access modifiers (like `public`, `private`, etc.) should have indentation level
  `0`.

- Don't indent namespaces.

- Pointer and reference symbols should be left aligned, e.g.

  .. code-block:: c++

      int* a;
      int& b;

- Operands of binary and ternary expressions, trailing comments should be
  horizontally aligned if
  they are split over multiple lines, e.g.:

  .. code-block:: c++

      int a = abc       // My comment
              * def;    // is quite long.
      bool aaaaa = bbbbbbbbbbbbbbbbbbbb
                   && ccccccccccccccccccccc;

- Don't add a newline before an opening curly bracket, e.g.:

  .. code-block:: c++

      void f(bool a) {
          if (a) {
              foo();
              bar();
          } else {
              try {
                  foo();
              } catch () {
              }
          }
      }

- Don't use a space before opening parenthesis, except after control statements,
  e.g.:

  .. code-block:: c++

      void foo(bool a) {
        if (a) {
          bar();
          for (auto& b : _c) {}
        }
      }

For more details, compare the `kotekan clang-format file
<https://github.com/kotekan/kotekan/blob/master/.clang-format>`_ and the
`formatting options of clang-format
<https://clang.llvm.org/docs/ClangFormatStyleOptions.html>`_


Automatic code formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_,
you can get free auto-formatting of that messy code you just wrote with
`make clang-format`.

So it doesn't happen again, you should check if there is `clang-format`
integration for your favourite editor and point it at kotekan's `.clang-format
file <https://github.com/kotekan/kotekan/blob/master/.clang-format>`_.


Disabling Code Formatting
 ^^^^^^^^^^^^^^^^^^^^^^^^^^
If you write a piece of code that you want to be excluded from auto-formatting,
you can prepend ``// clang-format off`` or ``/* clang-format off */`` and append
``// clang-format on`` or ``/* clang-format on */``.
