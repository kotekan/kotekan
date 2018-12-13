C++ Code Guidelines
---------------------

Naming
^^^^^^^^^^
- Type names in kotekan should be nouns, use *CamelCase* formatting and begin
  with a capital letter, e.g. ``MyFavouriteProcess``. This includes classes,
  enums, structs and typedefs.
- With the exception of class constructors & destructors, function names should
  use underscore notation and begin with a lower case letter, e.g. ``my_func``.

Variables
^^^^^^^^^^
Variables in the code should use underscore naming, e.g.
``my_favourite_variable``.

Explicit typing should be used wherever possible, e.g. always use ``uint32_t``
rather than ``uint``.

Private member variables should start with an underscore, e.g.
``_my_private_variable``.

Namespaces
^^^^^^^^^^
Avoid `using namespace X;`. Instead specify where you are using classes or
functions from a namespace, e.g. `std::vector<std::string> my variable;`.
Never do `using namespace std;`.

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

 - Header file implementing the interface of this `.cpp` file.
 - kotekan project headers
 - Local/Private Headers
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

      int a = abc *     // My comment
              def;      // is quite long.
      bool aaaaa = bbbbbbbbbbbbbbbbbbbb
                   && ccccccccccccccccccccc;

- The parameters in a function definition or declaration as well as the
  arguments in a function call should either be all in one line or in one line
  each and horizontally aligned, e.g.:

  .. code-block:: c++

      int a = f(b, c, d, e);
      int f(int b, int c, int d, int e);
      int f(int b, int c, int d, int e) {}

  .. code-block:: c++

      int a = f(
          bbb, ccc, ddd, eee);
      int f(
          int bbb, int ccc, int ddd, int edd);
      int f(
          int bbb, int ccc, int ddd, int eee) {}

  .. code-block:: c++

      int a = f(bbbbbb,
                cccccc,
                dddddd,
                eeeeee);
      int f(int bbbbbb,
            int cccccc,
            int dddddd,
            int eeeeee);
      int f(int bbbbbb,
            int cccccc,
            int dddddd,
            int eeeeee) {}

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

For more details, compare the `kotekan clang-format file
<https://github.com/kotekan/kotekan/blob/master/.clang-format>`_ and the
`formatting options of clang-format
<https://clang.llvm.org/docs/ClangFormatStyleOptions.html>`_