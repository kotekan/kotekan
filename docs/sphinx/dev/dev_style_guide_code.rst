C++ Code Guidelines
---------------------

For everything design related, you should follow the `C++ Core Guidelines
<http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines>`_. Additionally
to that, the following rules mostly describe the code formatting used in this
project.

Naming
^^^^^^^^^^
- Type names in kotekan should be nouns, use *PascalCase* formatting and begin
  with a capital letter, e.g. ``MyFavouriteStage``. This includes classes,
  enums, structs and typedefs.
- With the exception of class constructors & destructors, names of class members
  and standalone functions should use underscore notation and begin with a lower
  case letter, e.g. ``my_func``.

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
ending ``.hpp``, i.e., ``MyClass.hpp``. If the file contains mainly standalone
functions with maybe a grab bag of useful small/utility classes, name it using
``snake_case``, for example: ``networking_functions.hpp``.

Header files should begin with a header file commit like

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


Python Code Formatting
----------------------

All python code in this project should be formatted accoring to the
`black code style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_. You can let black take care
of that using ``black --exclude docs kotekan/root/dir`` before you commit python code.


Automatic code formatting
-------------------------

.. note:: To do all code linting in one, use `tools/lint.sh <https://github.com/kotekan/kotekan/blob/master/tools/lint.sh>`_. To use the script as a commit hook, copy it to ``.git/hooks/pre-commit``.

kotekan uses the following tools for automatic code formatting:

* `black <black.readthedocs.io>`_: python code formatting. Run it on the kotekan code base with ``black --exclude docs .``.
* `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_: C/C++ code formatting. Run it on your code with ``make clang-format`` from the build directory.
* `iwyu>=0.10 <include-what-you-use.org>`_: include-what-you-use for C/C++. Checks if source files include everything they use and nothing else. Run it on the code by ``cmake -DIWYU=ON .. && make > fix_include --nosafe_headers --comments``.

.. topic:: Disabling C/C++ Code Formatting

    If you write a piece of code that you want to be excluded from auto-formatting,
    you can prepend ``// clang-format off`` or ``/* clang-format off */`` and append
    ``// clang-format on`` or ``/* clang-format on */``.
